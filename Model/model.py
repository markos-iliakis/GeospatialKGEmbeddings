import re

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
        self.enc = enc
        self.path_dec = path_dec
        self.inter_dec = inter_dec
        self.inter_attn = inter_attn
        self.graph = graph
        self.cos = nn.CosineSimilarity(dim=0)
        self.use_inter_node = use_inter_node

    def forward(self, formula, queries, source_nodes, do_modelTraining=False):
        """
        Args:
            formula: a Formula() object
            queries: a list of Query() objects with the same formula
            source_nodes: a list of target node for each query (Training), a list of negative sampling nodes (query inferencing)
            do_modelTraining: default is False, we do query inferencing
        return:
            scores: a list of cosine scores with length len(queries)
        """
        if formula.query_type in ["1-chain", "2-chain", "3-chain"]:

            reverse_rels = tuple([self.graph._reverse_relation(formula.rels[i]) for i in range(len(formula.rels) - 1, -1, -1)])

            return self.path_dec.forward(
                self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_types[0]),
                self.enc.forward(source_nodes, formula.target_type), reverse_rels)

        elif formula.query_type == "3-inter_chain":
            target_embeds = self.enc(source_nodes, formula.target_type)

            # project the 1st anchor node to target node in rels: (t, p1, a1)
            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_types[0])
            embeds1 = self.path_dec.project(embeds1, self.graph._reverse_relation(formula.rels[0]))

            # project the 2nd anchor node to target node in rels:
            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_types[1])
            # '3-inter_chain': project a2 to t by following ((t, p2, e1),(e1, p3, a2))
            for i_rel in formula.rels[1][::-1]:  # loop the formula.rels[1] in the reverse order
                embeds2 = self.path_dec.project(embeds2, self.graph._reverse_relation(i_rel))

            query_intersection, embeds_inter = self.inter_dec([embeds1, embeds2])

            # Apply Graph Attention
            if self.use_inter_node and do_modelTraining:
                # for 2-inter, 3-inter, 3-inter_chain, the inter node is target node
                # we can use source_nodes
                query_embeds = self.graph.features([query.target_node for query in queries],
                                                   formula.target_type).t()
                query_intersection = self.inter_attn(query_embeds, embeds_inter, formula.target_type)
            else:
                query_intersection = self.inter_attn(query_intersection, embeds_inter, formula.target_type)

            scores = self.cos(target_embeds, query_intersection)
            return scores
        elif formula.query_type == "3-chain_inter":
            target_embeds = self.enc(source_nodes, formula.target_type)

            # project the 1st anchor node to inter node (e1, p2, a1)
            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_types[0])
            embeds1 = self.path_dec.project(embeds1, self.graph._reverse_relation(formula.rels[1][0]))

            # project the 2nd anchor node to inter node (e1, p3, a2)
            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_types[1])
            embeds2 = self.path_dec.project(embeds2, self.graph._reverse_relation(formula.rels[1][1]))

            # intersect different inter node (e1) embeddings
            query_intersection, embeds_inter = self.inter_dec([embeds1, embeds2])

            # Apply Graph Attention
            if self.use_inter_node and do_modelTraining:
                # for 3-chain_inter, the inter node is in the query_graph
                inter_nodes = [query.query_graph[1][2] for query in queries]
                inter_node_type = formula.rels[0][2]
                query_embeds = self.graph.features(inter_nodes, inter_node_type).t()
                query_intersection = self.inter_attn(query_embeds, embeds_inter, formula.target_type)
            else:
                query_intersection = self.inter_attn(query_intersection, embeds_inter, formula.rels[0][-1])

            query_intersection = self.path_dec.project(query_intersection, self.graph._reverse_relation(formula.rels[0]))
            scores = self.cos(target_embeds, query_intersection)
            return scores
        elif formula.query_type in ["2-inter", "3-inter"]:
            # x-inter
            target_embeds = self.enc(source_nodes, formula.target_type)
            num_edges = int(formula.query_type.replace("-inter", ""))
            embeds_list = []
            for i in range(0, num_edges):
                # project the ith anchor node to target node in rels: (t, pi, ai)
                embeds = self.enc([query.anchor_nodes[i] for query in queries], formula.anchor_types[i])
                embeds = self.path_dec.project(embeds, self.graph._reverse_relation(formula.rels[i]))
                embeds_list.append(embeds)

            query_intersection, embeds_inter = self.inter_dec(embeds_list)

            # Apply Graph Attention
            if self.use_inter_node and do_modelTraining:
                # for x-inter, the inter node is target node
                # so we can use the real target node
                query_embeds = self.graph.features([query.target_node for query in queries],
                                                   formula.target_type).t()
                query_intersection = self.inter_attn(query_embeds, embeds_inter, formula.target_type)
            else:
                # print ("DEBUG: ", query_intersection, embeds_inter)
                query_intersection = self.inter_attn(query_intersection, embeds_inter, formula.target_type)
                # print ("Done that !")

            scores = self.cos(target_embeds, query_intersection)
            return scores

    def margin_loss(self, formula, queries, hard_negatives=False, margin=1):
        if "inter" not in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_type]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries, [query.target_node for query in queries], do_modelTraining=True)
        neg_affs = self.forward(formula, queries, neg_nodes, do_modelTraining=True)
        # affs (neg_affs) is the cosine similarity between golden (negative) node embedding and predicted embedding
        # the larger affs (the smaller the neg_affs), the better the prediction is
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss
