import math
import numpy as np
import torch
from numpy import random
from torch import nn


def geo_lookup(nodes, id2geo, add_dim=-1, id2extent=None, do_extent_sample=False):
    """
    Given a list of node id, make a coordinate tensor
    Args:
        :param do_extent_sample: take random point in bounding box
        :param id2extent: a dict(): node id -> (xmin, xmax, ymin, ymax)
        :param id2geo: a dict(): node id -> [longitude, latitude]
        :param nodes: list of nodes id
        :param add_dim: add a dimension
    Return:
        coord_tensor: [batch_size, 2], geographic coordinate for geo_ent

    """
    coord_tensor = []
    for i, eid in enumerate(nodes):
        if eid in id2extent:
            if do_extent_sample:
                xmin, xmax, ymin, ymax = id2extent[eid]
                x = random.uniform(xmin, xmax)
                y = random.uniform(ymin, ymax)
                coords = [x, y]
            else:
                coords = list(id2geo[eid])
        else:
            coords = list(id2geo[eid])
        if add_dim == -1:
            coord_tensor.append(coords)
        elif add_dim == 1:
            coord_tensor.append([coords])

    return coord_tensor


""" Feature Encoders """


class DirectEncoder(nn.Module):
    """
    Encodes a node as an embedding via direct lookup.
    (i.e., this is just like basic node2vec or matrix factorization)
    """

    def __init__(self, features, feature_modules):
        """
        Initializes the model for a specific graph.

        features(nodes, mode): an embedding lookup function to make a dict() from node type to EmbeddingBag
            nodes: a lists of global node id which are in type (mode)
            mode: node type
            return: embedding vectors, shape [num_node, embed_dim]
        feature_modules: a dict of embedding matrix by node type, each embed matrix shape: [num_ent_by_type + 2, embed_dim]
        """
        super(DirectEncoder, self).__init__()
        for name, module in feature_modules.items():
            self.add_module("feat-" + name, module)
        self.features = features

    def forward(self, nodes, n_type, **kwargs):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        n_type    -- string designating the type of the nodes

        Returns a dict() map: node type --> embedding tensor [num_ent, embed_dim]
        """

        # Transpose embedding tensor as [embed_dim, num_ent]
        embeds = self.features(nodes, n_type).t()

        # Calculate the L2-norm for each embedding vector, [1, num_ent]
        norm = embeds.norm(p=2, dim=0, keepdim=True)

        # normalize the embedding vectors
        # shape: [embed_dim, num_ent]
        return embeds.div(norm)


""" Position Encoders """


class TheoryGridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """

    def __init__(self, spa_embed_dim, coord_dim=2, frequency_num=16,
                 max_radius=10000, min_radius=1000, freq_init="geometric",
                 ffn=None, device="cpu"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimension
            coord_dim: the dimension of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(TheoryGridCellSpatialRelationEncoder, self).__init__()
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.spa_embed_dim = spa_embed_dim
        self.freq_init = freq_init

        # the frequency we use for each block, alpha in ICLR paper
        self.freq_list = self.cal_freq_list()
        self.freq_mat = self.cal_freq_mat()

        # there unit vectors which is 120 degree apart from each other
        self.unit_vec1 = np.asarray([1.0, 0.0])  # 0
        self.unit_vec2 = np.asarray([-1.0 / 2.0, math.sqrt(3) / 2.0])  # 120 degree
        self.unit_vec3 = np.asarray([-1.0 / 2.0, -math.sqrt(3) / 2.0])  # 240 degree

        self.input_embed_dim = self.cal_input_dim()

        self.ffn = ffn

        self.device = device

    def cal_freq_list(self):
        log_timescale_increment = (math.log(float(self.max_radius) / float(self.min_radius)) /
                                   (self.frequency_num * 1.0 - 1))

        timescales = self.min_radius * np.exp(
            np.arange(self.frequency_num).astype(float) * log_timescale_increment)

        return 1.0 / timescales

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)

        # self.freq_mat shape: (frequency_num, 6)
        return np.repeat(freq_mat, 6, axis=1)

    def cal_input_dim(self):
        # compute the dimension of the encoded spatial relation embedding
        return int(6 * self.frequency_num)

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # compute the dot product between [deltaX, deltaY] and each unit_vec
        # (batch_size, num_context_pt, 1)
        angle_mat1 = np.expand_dims(np.matmul(coords_mat, self.unit_vec1), axis=-1)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = np.expand_dims(np.matmul(coords_mat, self.unit_vec2), axis=-1)
        # (batch_size, num_context_pt, 1)
        angle_mat3 = np.expand_dims(np.matmul(coords_mat, self.unit_vec3), axis=-1)

        # (batch_size, num_context_pt, 6)
        angle_mat = np.concatenate([angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3], axis=-1)
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = np.expand_dims(angle_mat, axis=-2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = np.repeat(angle_mat, self.frequency_num, axis=-2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = angle_mat * self.freq_mat
        # (batch_size, num_context_pt, frequency_num*6)
        spr_embeds = np.reshape(angle_mat, (batch_size, num_context_pt, -1))

        # make sinusoid function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, frequency_num*6=input_embed_dim)
        spr_embeds[:, :, 0::2] = np.sin(spr_embeds[:, :, 0::2])  # dim 2i
        spr_embeds[:, :, 1::2] = np.cos(spr_embeds[:, :, 1::2])  # dim 2i+1

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        return self.ffn(spr_embeds)


class ExtentPositionEncoder(nn.Module):
    """
    This is position encoder, a wrapper for different space encoders,
    Given a list of node ids, return their embedding
    """

    def __init__(self, id2geo, id2extent, spa_enc, graph, device="cpu"):
        """
        Args:
            out_dims: a dict()
                key: node type
                value: embedding dimension

            id2geo: a dict(): node id -> [longitude, latitude]
            id2extent: a dict(): node id -> (xmin, xmax, ymin, ymax)
            spa_enc: one space encoder
            graph: Graph()
        """
        super(ExtentPositionEncoder, self).__init__()
        self.id2geo = id2geo
        self.id2extent = id2extent
        self.spa_embed_dim = spa_enc.spa_embed_dim  # the output space embedding
        self.spa_enc = spa_enc
        self.graph = graph
        self.device = device

    def forward(self, nodes):
        """
        Args:
            nodes: a list of node ids
        Return:
            pos_embeds: the position embedding for all nodes, (spa_embed_dim, batch_size)
                    geo_ent => space embedding from geographic coordinates
        """
        # coord_tensor: [batch_size, 1, 2], geographic coordinate for geo_ent
        coord_tensor = geo_lookup(nodes, id2geo=self.id2geo, add_dim=1, id2extent=self.id2extent, do_extent_sample=True)

        # spa_embeds: (batch_size, 1, spa_embed_dim)
        spa_embeds = self.spa_enc(coord_tensor)

        # spa_embeds: (batch_size, spa_embed_dim)
        pos_embeds = torch.squeeze(spa_embeds, dim=1)

        # pos_embeds: (spa_embed_dim, batch_size)
        pos_embeds = pos_embeds.t()

        return pos_embeds


""" Wrapper Encoder """


class NodeEncoder(nn.Module):
    """
    This is the encoder for each entity or node which has two components"
    1. feature encoder (DirectEncoder): feat_enc
    2. position encoder (PositionEncoder): pos_enc
    """

    def __init__(self, feat_enc, pos_enc, agg_type="add"):
        '''
        Args:
            feat_enc:feature encoder
            pos_enc: position encoder
            agg_type: how to combine the feature embedding and space embedding of a node/entity
        '''
        super(NodeEncoder, self).__init__()
        self.feat_enc = feat_enc
        self.pos_enc = pos_enc
        self.agg_type = agg_type
        if feat_enc is None and pos_enc is None:
            raise Exception("pos_enc and feat_enc are both None!!")

    def forward(self, nodes, mode, offset=None):
        '''
        Args:
            nodes: a list of node ids
        Return:

            embeds: node embedding
                if agg_type in ["add", "min", "max", "mean"]:
                    # here we assume spa_embed_dim == embed_dim
                    shape [embed_dim, num_ent]
                if agg_type == "concat":
                    shape [embed_dim + spa_embed_dim, num_ent]
        '''
        if self.feat_enc is not None and self.pos_enc is not None:
            # we have both feature encoder and position encoder

            # feat_embeds: [embed_dim, num_ent]
            feat_embeds = self.feat_enc(nodes, mode, offset=offset)

            # pos_embeds: [embed_dim, num_ent]
            pos_embeds = self.pos_enc(nodes)
            if self.agg_type == "add":
                embeds = feat_embeds + pos_embeds
            elif self.agg_type in ["min", "max", "mean"]:

                if self.agg_type == "min":
                    agg_func = torch.min
                elif self.agg_type == "max":
                    agg_func = torch.max
                elif self.agg_type == "mean":
                    agg_func = torch.mean
                combined = torch.stack([feat_embeds, pos_embeds])
                aggs = agg_func(combined, dim=0)
                if type(aggs) == tuple:
                    aggs = aggs[0]
                embeds = aggs
            elif self.agg_type == "concat":
                embeds = torch.cat([feat_embeds, pos_embeds], dim=0)
            else:
                raise Exception("The Node Encoder Aggregation type is not supported!!")
        elif self.feat_enc is None and self.pos_enc is not None:
            # we only have position encoder

            # pos_embeds: [embed_dim, num_ent]
            pos_embeds = self.pos_enc(nodes)

            embeds = pos_embeds
        elif self.feat_enc is not None and self.pos_enc is None:
            # we only have feature encoder

            # feat_embeds: [embed_dim, num_ent]
            feat_embeds = self.feat_enc(nodes, mode, offset=offset)

            embeds = feat_embeds

        return embeds
