import torch
from torch import nn
import torch.nn.functional as F
from Model.MultiLayerFFN import LayerNorm


""" Decoder module which takes pairs of embeddings and predicts relationship scores given these embeddings. """


class BilinearBlockDiagMetapathDecoder(nn.Module):
    """
    This is only used for enc_agg_func == "concat"
    Each edge type is represented by two matrices:
    1) feature matrix for node feature embed
    2) position matrix for node position embed
    It can be seen as a block-diag matrix
    compositional relationships are a product matrices.

    The forward method returns a compositional relationships score,
    i.e. the likelihood of compositional relationship or metapath, between a pair of nodes.
    """

    def __init__(self, relations, dims, feat_dims, spa_embed_dim):
        """
        Args:
            relations: a dict() of all triple templates
                key:    domain entity type
                value:  a list of tuples (range entity type, predicate)
            feat_dims: a dict(), node type => embed_dim of feature embedding
            spa_embed_dim: the embed_dim of position embedding
        """
        super(BilinearBlockDiagMetapathDecoder, self).__init__()
        self.relations = relations
        self.feat_dims = feat_dims
        self.spa_embed_dim = spa_embed_dim

        self.feat_mats = {}
        self.pos_mats = {}
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=0)
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.feat_mats[rel] = nn.Parameter(torch.FloatTensor(feat_dims[rel[0]], feat_dims[rel[2]]))
                nn.init.xavier_uniform_(self.feat_mats[rel])
                self.register_parameter("feat-" + "_".join(rel), self.feat_mats[rel])

                self.pos_mats[rel] = nn.Parameter(torch.FloatTensor(spa_embed_dim, spa_embed_dim))
                nn.init.xavier_uniform_(self.pos_mats[rel])
                self.register_parameter("pos-" + "_".join(rel), self.pos_mats[rel])

    def forward(self, embeds1, embeds2, rels):
        """
        embeds1, embeds2 shape: [embed_dim, batch_size]
        rels: a list of triple templates, an n-length metapath
        """
        # act: [batch_size, embed_dim]
        act = embeds1.t()
        feat_act, pos_act = torch.split(act, [self.feat_dims[rels[0][0]], self.spa_embed_dim], dim=1)
        for i_rel in rels:
            feat_act = feat_act.mm(self.feat_mats[i_rel])
            pos_act = pos_act.mm(self.pos_mats[i_rel])
        #  act: [batch_size, embed_dim]
        act = torch.cat([feat_act, pos_act], dim=1)
        act = self.cos(act.t(), embeds2)
        return act

    def project(self, embeds, rel):
        """
        embeds shape: [embed_dim, batch_size]
        rel: triple template
        """
        feat_act, pos_act = torch.split(embeds.t(),
                                        [self.feat_dims[rel[0]], self.spa_embed_dim], dim=1)
        feat_act = feat_act.mm(self.feat_mats[rel])
        pos_act = pos_act.mm(self.pos_mats[rel])
        act = torch.cat([feat_act, pos_act], dim=1)
        return act.t()


class BoxDecoder(nn.Module):

    def __init__(self, relations, feat_embed_dim, spa_embed_dim):
        """
       Args:
           relations: a dict() of all triple templates
               key:    domain entity type
               value:  a list of tuples (range entity type, predicate)
           feat_embed_dim: the embed_dim of feature embedding
           spa_embed_dim: the embed_dim of position embedding
       """
        super(BoxDecoder, self).__init__()
        self.relations = relations
        self.feat_embed_dim = feat_embed_dim
        self.spa_embed_dim = spa_embed_dim

        self.feat_mats = {}
        self.pos_mats = {}
        self.rel_offset_embeddings = {}
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=0)
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.feat_mats[rel] = nn.Parameter(torch.FloatTensor(feat_embed_dim, feat_embed_dim))
                nn.init.xavier_uniform_(self.feat_mats[rel])
                self.register_parameter("feat-" + "_".join(rel), self.feat_mats[rel])

                self.pos_mats[rel] = nn.Parameter(torch.FloatTensor(spa_embed_dim, spa_embed_dim))
                nn.init.xavier_uniform_(self.pos_mats[rel])
                self.register_parameter("pos-" + "_".join(rel), self.pos_mats[rel])

                self.rel_offset_embeddings[rel] = nn.Parameter(torch.FloatTensor(feat_embed_dim + spa_embed_dim, feat_embed_dim + spa_embed_dim))
                nn.init.xavier_uniform_(self.rel_offset_embeddings[rel])
                self.register_parameter('off-' + '_'.join(rel), self.rel_offset_embeddings[rel])

    def forward(self, embeddings, offset_embeddings, rel):
        """
        embeds shape: [embed_dim, batch_size]
        rel: triple template
        """
        # Create the relation embedding by concatenating the feature and position embeddings
        rel_embeddings = torch.cat([self.feat_mats[rel], self.pos_mats[rel]])

        # Add up the centers
        embeddings += rel_embeddings

        # Add up the offsets
        offset_embeddings += self.sigmoid(self.rel_offset_embeddings)

        return embeddings.t(), offset_embeddings.t()


""" Intersection Operator """


class SimpleSetIntersection(nn.Module):
    """
    Decoder that computes the implicit intersection between two state vectors.
    Takes a simple element-wise min.
    """
    def __init__(self, agg_func=torch.min):
        super(SimpleSetIntersection, self).__init__()
        self.agg_func = agg_func

    def forward(self, embeds_list):
        if len(embeds_list) < 2:
            raise Exception("The intersection needs more than one embedding")

        combined = torch.stack(embeds_list)
        aggs = self.agg_func(combined, dim=0)
        if type(aggs) == tuple:
            aggs = aggs[0]
        return aggs, combined


class BoxOffsetIntersection(nn.Module):

    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


class BoxCenterIntersectAttention(nn.Module):

    def __init__(self, out_dims, types, num_attn):
        super(BoxCenterIntersectAttention, self).__init__()
        self.out_dims = out_dims
        self.num_attn = num_attn
        self.activation = nn.LeakyReLU
        self.f_activation = nn.Sigmoid
        self.softmax = nn.Softmax(dim=0)

        self.attn_vecs = {}
        self.norms = {}

        for type in types:
            self.norms[type] = LayerNorm(out_dims)
            self.add_module(type + "_ln", self.norms[type])

            # each column represent an attention vector for one attention head: [embed_dim*2, num_attn]
            self.atten_vecs[type] = nn.Parameter(torch.FloatTensor(2*out_dims, self.num_attn))
            nn.init.xavier_uniform_(self.atten_vecs[type])
            self.register_parameter(type + "_attenvecs", self.atten_vecs[type])

    def forward(self, embeddings, type):
        attention = torch.einsum("nbd,dk->nbk", (embeddings, self.atten_vecs[type]))
        attention = self.activation(attention)
        attention = self.softmax(attention)

        combined = torch.einsum("bkn,bnd->bkd", (attention, embeddings))
        combined = self.f_activation(combined)
        combined = combined + embeddings
        combined = self.norms[type](combined)
        return combined
        

""" Attention method used for query attention learning """


class IntersectConcatAttention(nn.Module):
    def __init__(self, query_dims, key_dims, activation, f_activation, num_attn=1, layernorm=False, use_post_mat=False):
        """
        The attention method used by Graph Attention network (LeakyReLU)
        Args:
            query_dims: a dict() mapping: node type --> pre-computed variable embeddings dimension
            key_dims: a dict() mapping: node type --> embeddings dimension computed from different query path for the same variables
            num_attn: number of attention heads
            activation: the activation function to atten_vecs * torch.cat(query_embed, key_embed), see GAT paper Equ 3
            f_activation: the final activation function applied to get the final result, see GAT paper Equ 6
        """
        super(IntersectConcatAttention, self).__init__()
        self.atten_vecs = {}
        self.query_dims = query_dims
        self.key_dims = key_dims
        self.num_attn = num_attn

        # define the layer normalization
        self.layernorm = layernorm
        if self.layernorm:
            self.lns = {}
            for type in query_dims:
                self.lns[type] = LayerNorm(query_dims[type])
                self.add_module(type + "_ln", self.lns[type])

        self.use_post_mat = use_post_mat
        if self.use_post_mat:
            self.post_W = {}
            self.post_B = {}
            if self.layernorm:
                self.post_lns = {}
            for type in query_dims:
                self.post_W[type] = nn.Parameter(torch.FloatTensor(query_dims[type], query_dims[type]))
                nn.init.xavier_uniform_(self.post_W[type])
                self.register_parameter(type + "_attnPostW", self.post_W[type])

                self.post_B[type] = nn.Parameter(torch.FloatTensor(query_dims[type], 1))
                nn.init.xavier_uniform_(self.post_B[type])
                self.register_parameter(type + "_attnPostB", self.post_B[type])
                if self.layernorm:
                    self.post_lns[type] = LayerNorm(query_dims[type])
                    self.add_module(type + "_attnPostln", self.post_lns[type])

        self.activation = activation

        self.f_activation = f_activation

        self.softmax = nn.Softmax(dim=0)

        for type in query_dims:
            # each column represent an attention vector for one attention head: [embed_dim*2, num_attn]
            self.atten_vecs[type] = nn.Parameter(torch.FloatTensor(query_dims[type] + key_dims[type], self.num_attn))
            nn.init.xavier_uniform_(self.atten_vecs[type])
            self.register_parameter(type + "_attenvecs", self.atten_vecs[type])

    def forward(self, query_embed, key_embeds, type):
        """
        Args:
            query_embed: the pre-computed variable embeddings, [embed_dim, batch_size]
            key_embeds: a list of embeddings computed from different query path for the same variables, [num_query_path, embed_dim, batch_size]
            type: node type
        Return:
            combined: the multi-head attention based embeddings for a variable [embed_dim, batch_size]
        """
        tensor_size = key_embeds.size()
        num_query_path = tensor_size[0]
        batch_size = tensor_size[2]

        # query_embed_expand: [num_query_path, embed_dim, batch_size]
        query_embed_expand = query_embed[0].unsqueeze(0).expand_as(key_embeds)

        # concat: [num_query_path, batch_size, embed_dim*2]
        concat = torch.cat((query_embed_expand, key_embeds), dim=1).transpose(1, 2)

        # 1. compute the attention score for each key embeddings
        # attn: [num_query_path, batch_size, num_attn]
        attn = torch.einsum("nbd,dk->nbk", (concat, self.atten_vecs[type]))

        # attn: [num_query_path, batch_size, num_attn]
        attn = self.softmax(self.activation(attn))

        # attn: [batch_size, num_attn, num_context_pt]
        attn = attn.transpose(0, 1).transpose(1, 2)

        # key_embeds_: [batch_size, num_query_path, embed_dim]
        key_embeds_ = key_embeds.transpose(1, 2).transpose(0, 1)

        # 2. using the attention score to compute the weighted average of the key embeddings
        # combined: [batch_size, num_attn, embed_dim]
        combined = torch.einsum("bkn,bnd->bkd", (attn, key_embeds_))

        # combined: [batch_size, embed_dim]
        combined = torch.sum(combined, dim=1, keepdim=False) * (1.0 / self.num_attn)

        # combined: [embed_dim, batch_size]
        combined = self.f_activation(combined).t()

        if self.layernorm:
            combined = combined + query_embed[0]
            combined = self.lns[type](combined.t()).t()

        if self.use_post_mat:
            linear = self.post_W[type].mm(combined) + self.post_B[type]
            if self.layernorm:
                linear = linear + combined
                linear = self.post_lns[type](linear.t()).t()
            return linear

        return combined
