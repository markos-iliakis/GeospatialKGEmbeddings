import torch.nn.functional as F
import torch
from numpy import random
from torch import nn

"""
End-to-end autoencoder models for representation learning on
heterogeneous graphs
"""


class QueryEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons about edges, meta-paths and intersections
    """

    def __init__(self, graph, enc, path_dec, inter_dec, inter_attn, use_inter_node=False):
        """
        Args:
            graph: the Graph() object
            enc: the node embedding encoder module
            path_dec: the metapath decoder module
            inter_dec: the intersection decoder module
            inter_attn: the intersection attention module
            use_inter_node: Whether we use the True nodes in the intersection attention as the query embedding to train the QueryEncoderDecoder
        """
        super(QueryEncoderDecoder, self).__init__()
        self.a = 0.02  # Hyperparameter that balances the in-box distance and the out-box distance
        self.gamma = nn.Parameter(torch.Tensor([24]), requires_grad=False)
        self.enc = enc
        self.path_dec = path_dec
        self.inter_dec = inter_dec
        self.inter_attn = inter_attn
        self.graph = graph
        self.cos = nn.CosineSimilarity(dim=0)
        self.use_inter_node = use_inter_node

    def forward(self, formula, queries, source_nodes, modelTraining=False):
        """
        Args:
            formula: a Formula() object
            queries: a list of Query() objects with the same formula
            source_nodes: a list of target node for each query (Training), a list of negative sampling nodes (query inferencing)
            modelTraining: default is False, we do query inferencing
        return:
            scores: a list of cosine scores with length len(queries)
        """
        if formula.query_type in ["1-chain", "2-chain", "3-chain"]:
            # o->t       ((t, p1, a1))
            # o->o->t    ((t, p1, e1),(e1, p2, a1))
            # o->o->o->t ((t, p1, e1),(e1, p2, e2),(e2, p3, a1))

            reverse_rels = tuple(
                [self.graph._reverse_relation(formula.rels[i]) for i in range(len(formula.rels) - 1, -1, -1)])
            return self.path_dec.forward(
                self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_types[0]),
                self.enc.forward(source_nodes, formula.target_type), reverse_rels)

        elif formula.query_type == "3-inter_chain":
            #        t<-o<-o  ((t, p2, e1), (e1, p3, a2))
            #        ^
            #        |
            #        o        (t, p1, a1)

            # Encode the target variable / target_embeds -> [embed_size, batch_size]
            target_embeds = self.enc(source_nodes, formula.target_type)

            # Encode the 1st anchor node / embeds -> [embed_size {128} -> [feat {64} + pos {64}], batch_size]
            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_types[0])

            # Project the 1st anchor node a1 through its relation p1 (t, p1, a1) creating the answer /embeds -> [embed_size {128}, batch_size]
            embeds1 = self.path_dec.project(embeds1, self.graph._reverse_relation(formula.rels[0]))

            # Encode the 2nd anchor node / embeds -> [embed_size {128} -> [feat {64} + pos {64}], batch_size]
            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_types[1])

            # Project the 2nd anchor node a2 and each inter node ei through its relation pi creating second answer box
            for i_rel in formula.rels[1][-1:0:-1]:  # loop the formula.rels[1] in the reverse order
                # embeds, offset_embeddings -> [embed_size {128}, batch_size]
                embeds2 = self.path_dec.project(embeds2, self.graph._reverse_relation(i_rel))

            # Intersect the 2 boxes / query_intersect_cen, query_intersection_off -> [batch_size, embed_dim]
            query_intersection, embeds_inter = self.inter_dec(formula.target_type, [embeds1, embeds2])
            if self.use_inter_node and modelTraining:
                # for 2-inter, 3-inter, 3-inter_chain, the inter node is target node
                # we we can use source_nodes
                query_embeds = self.graph.features([query.target_node for query in queries],
                                                   formula.target_type).t()
                query_intersection = self.inter_attn(query_embeds, embeds_inter, formula.target_type)
            else:
                query_intersection = self.inter_attn(query_intersection, embeds_inter, formula.target_type)

            # Return the cosine similarity of the target embedding with the predicted embedding
            return self.cos(target_embeds, query_intersection)

        elif formula.query_type == "3-chain_inter":
            #        t      (t, p1, e1)
            #        ^
            #        |
            #     o->o<-o   ((e1, p2, a1), (e1, p3, a2))

            # Encode the target variable / target_embeds -> [embed_size, batch_size]
            target_embeds = self.enc(source_nodes, formula.target_type)

            # Encode the 1st anchor node / embeds -> [embed_size {128} -> [feat {64} + pos {64}], batch_size]
            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_types[0])

            # Project the 1st anchor node a1 through its relation p2 to inter node e1 (e1, p2, a1) creating a box /embeds, offset_embeddings -> [embed_size {128}, batch_size]
            embeds1 = self.path_dec.project(embeds1, self.graph._reverse_relation(formula.rels[1][0]))

            # Encode the 2nd anchor node / embeds -> [embed_size {128} -> [feat {64} + pos {64}], batch_size]
            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_types[1])

            # Project the 2nd anchor node a2 through its relation p3 to inter node e1 (e1, p3, a2) creating a second box /embeds, offset_embeddings -> [embed_size {128}, batch_size]
            embeds2 = self.path_dec(embeds2, self.graph._reverse_relation(formula.rels[1][1]))

            # intersect the 2 boxes / query_intersect_cen, query_intersection_off -> [batch_size, embed_dim]
            query_intersection, embeds_inter = self.inter_dec(formula.rels[0][-1], [embeds1, embeds2])
            if self.use_inter_node and modelTraining:
                # for 3-chain_inter, the inter node is in the query_graph
                inter_nodes = [query.query_graph[1][2] for query in queries]
                inter_node_mode = formula.rels[0][2]
                query_embeds = self.graph.features(inter_nodes, inter_node_mode).t()
                query_intersection = self.inter_attn(query_embeds, embeds_inter, formula.target_type)
            else:
                query_intersection = self.inter_attn(query_intersection, embeds_inter, formula.rels[0][-1])

            # Project the inter node e1 through its relation p1 to the target node creating the answer box
            query_intersection = self.path_dec.project(query_intersection, self.graph._reverse_relation(formula.rels[0]))

            # Return the distance of the target from the box
            return self.cos(target_embeds, query_intersection)

        elif formula.query_type in ["2-inter", "3-inter"]:
            # o->t<-o  ((t, p1, a1),(t, p2, a2))     o->t<-o  ((t, p1, a1),(t, p2, a2),(t, p3, a3))
            #                                           ^
            #                                           |
            #                                           o

            # Encode the target variable / target_embeds -> [embed_size, batch_size]
            target_embeds = self.enc(source_nodes, formula.target_type)

            # Count the number of edges
            num_edges = int(formula.query_type.replace("-inter", ""))

            embeds_list = []
            offset_list = []
            # For each anchor node ai project it through its relation pi to the target node creating an answer box
            for i in range(0, num_edges):
                # Encode the anchor node / embeds -> [embed_size {128} -> [feat {64} + pos {64}], batch_size]
                embeds = self.enc([query.anchor_nodes[i] for query in queries], formula.anchor_types[i])

                # Project the anchor node ai through its relation pi to the target node creating an answer box /embeds, offset_embeddings -> [embed_size {128}, batch_size]
                embeds = self.path_dec.project(embeds, self.graph._reverse_relation(formula.rels[i]))

                embeds_list.append(embeds)

            # Intersect the boxes / query_intersect_cen, query_intersection_off -> [batch_size, embed_dim]
            query_intersection, embeds_inter = self.inter_dec(formula.target_type, embeds_list)
            if self.use_inter_node and modelTraining:
                # for x-inter, the inter node is target node
                # so we can use the real target node
                query_embeds = self.graph.features([query.target_node for query in queries], formula.target_type).t()
                query_intersection = self.inter_attn(query_embeds, embeds_inter, formula.target_type)
            else:
                query_intersection = self.inter_attn(query_intersection, embeds_inter, formula.target_type)

            # Return the distance of the target from the box
            return self.cos(target_embeds, query_intersection)

    def dist_box(self, entity_embedding, query_center_embedding, query_offset_embedding):
        entity_embedding = entity_embedding.t()

        delta = (entity_embedding - query_center_embedding).abs()

        # Use ReLU for d_out to zero the negatives in the vector
        distance_out = F.relu(delta - query_offset_embedding)

        distance_in = torch.min(delta, query_offset_embedding)

        # Downweight the distance inside the box as we regard entities inside the box close enough to the center
        distance_box = torch.norm(distance_out, p=1, dim=-1) + self.a * torch.norm(distance_in, p=1, dim=-1)

        # logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.a * torch.norm(distance_in, p=1, dim=-1)
        return distance_box

    def box_loss(self, formula, queries, hard_negatives=False):
        if hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_type]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        positive_dist_box = self.forward(formula, queries, [query.target_node for query in queries])
        negative_dist_box = self.forward(formula, queries, neg_nodes)

        negative_score = F.logsigmoid(negative_dist_box - self.gamma)  # .mean() if negatives more than 1/query
        positive_score = F.logsigmoid(self.gamma - positive_dist_box)
        loss = -positive_score - negative_score

        return loss.sum()

    def margin_loss(self, formula, queries, hard_negatives=False, margin=1):
        if "inter" not in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_type]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries, [query.target_node for query in queries], modelTraining=True)
        neg_affs = self.forward(formula, queries, neg_nodes, modelTraining=True)
        # affs (neg_affs) is the cosine similarity between golden (negative) node embedding and predicted embedding
        # the larger affs (the smaller the neg_affs), the better the prediction is
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss
