import numpy as np
import random
import torch
from scipy.stats import stats
from sklearn.metrics import roc_auc_score

from Model.MultiLayerFFN import MultiLayerFeedForwardNN
from Model.decoder import BilinearBlockDiagMetapathDecoder, SimpleSetIntersection, IntersectConcatAttention, BoxDecoder, \
    BoxOffsetIntersection, CenterIntersection, BoxCenterIntersectAttention
from Model.encoder import DirectEncoder, TheoryGridCellSpatialRelationEncoder, ExtentPositionEncoder, NodeEncoder
from Model.model import QueryEncoderDecoder
from Yago2GeoDatasetHelpers.create_data_files import unite_files, make_id_files, make_custom_triples, \
    custom_triples_split, make_graph
from Yago2GeoDatasetHelpers.dataset_helper_functions import read_id2geo, read_id2extent
from Yago2GeoDatasetHelpers.query_sampling import make_single_edge_query_data, sample_new_clean, \
    make_multiedge_query_data, make_inter_query_data


def create_paths():
    triples_path = '../yago2geo/triples/materialized/'
    types_geo_path = '../yago2geo/geo_classes/'

    data_path = 'Data/'
    classes_path = data_path + 'entity2type.json'
    entitiesID_path = data_path + 'entity2id.txt'
    graph_path = data_path + 'graph.pkl'
    train_path = data_path + 'train_triples.pkl'
    valid_path = data_path + 'valid_triples.pkl'
    test_path = data_path + 'test_triples.pkl'

    return data_path, classes_path, entitiesID_path, graph_path, triples_path, types_geo_path, train_path, valid_path, test_path


def create_data(data_path, classes_path, entitiesID_path, graph_path, triples_path, types_geo_path):
    new_triples_path = data_path + 'united_triples.nt'
    rid2inverse_path = data_path + 'rid2inverse.json'
    relationsID_path = data_path + 'relation2id.txt'
    id2geo_path = data_path + 'id2geo.json'

    # Unite all triples
    unite_files(triples_path, types_geo_path, data_path)

    # Make the entity2id, relation2id with inverses, relations2inverse and relationsID2Inverse files
    make_id_files(new_triples_path, types_geo_path, data_path)

    # Make the triples in the form of (subject_id, (subject_type, relation, object_type), object_id)
    custom_triples = make_custom_triples(new_triples_path, classes_path, relationsID_path, entitiesID_path, data_path)

    # Split the triples to train / valid / test and put them to pkl files
    custom_triples_split(custom_triples, data_path)

    # Make the graph file
    make_graph(custom_triples, classes_path, entitiesID_path, graph_path, rid2inverse_path)

    # Make train / valid / test 1-chain queries
    make_single_edge_query_data(data_path, graph_path, 100)

    # Make train / valid / test 2/3-chain queries
    mp_result_dir = data_path + 'train_queries_mp/'
    sample_new_clean(data_path, graph_path)
    make_multiedge_query_data(data_path, graph_path, 50, 20000, mp_result_dir=mp_result_dir)
    # print('keeping low samples and workers for testing')
    # make_multiedge_query_data(data_path, graph_path, 1, 1, mp_result_dir=mp_result_dir)

    # Make train x-inter queries
    mp_result_dir = data_path + 'train_inter_queries_mp/'
    make_inter_query_data(data_path, graph_path, 50, 10000, max_inter_size=7, mp_result_dir=mp_result_dir)
    # print('keeping low samples and workers for testing')
    # make_inter_query_data(data_path, graph_path, 1, 1, max_inter_size=7, mp_result_dir=mp_result_dir)

    # Make valid/testing 2/3 edges geographic queries, negative samples are geo-entities
    id2geo = read_id2geo(id2geo_path)
    sample_new_clean(data_path, graph_path, id2geo=id2geo)

    # Make train x-inter queries, negative samples are geo-entities
    print("Do geo content sample")
    mp_result_geo_dir = data_path + "train_inter_queries_geo_mp/"
    id2geo = read_id2geo(id2geo_path)
    make_inter_query_data(data_path, graph_path, 50, 10000, max_inter_size=7, mp_result_dir=mp_result_geo_dir, id2geo=id2geo)

    mp_result_geo_dir = data_path + "train_queries_geo_mp/"
    id2geo = read_id2geo(id2geo_path)
    make_multiedge_query_data(data_path, graph_path, 50, 20000, mp_result_dir=mp_result_geo_dir, id2geo=id2geo)


def create_architecture(graph, feature_modules, feat_embed_dim, spa_embed_dim, do_train):
    out_dims = feat_embed_dim + spa_embed_dim
    types = [type for type in graph.relations]

    print('Creating Encoder Operator..')
    # encoder
    feat_enc = DirectEncoder(graph.features, feature_modules)
    ffn = MultiLayerFeedForwardNN(input_dim=6 * 16, output_dim=64, num_hidden_layers=1, dropout_rate=0.5, hidden_dim=512, activation='sigmoid', use_layernormalize=True, skip_connection=True)
    spa_enc = TheoryGridCellSpatialRelationEncoder(spa_embed_dim=64, coord_dim=2, frequency_num=16, max_radius=5400000, min_radius=50, freq_init='geometric', ffn=ffn, device='cuda')
    pos_enc = ExtentPositionEncoder(id2geo=read_id2geo('./Data/id2geo.json'), id2extent=read_id2extent('./Data/id2geo.json'), spa_enc=spa_enc, graph=graph, device='cuda')
    enc = NodeEncoder(feat_enc, pos_enc, agg_type='concat')

    print('Creating Projection Operator..')
    # decoder-projection
    dec = BoxDecoder(graph.relations, feat_embed_dim=feat_embed_dim, spa_embed_dim=spa_embed_dim)

    print('Creating Intersection Operator..')
    # intersection-attention
    # inter_dec_cen = CenterIntersection(dim=out_dims)
    inter_dec_cen = BoxCenterIntersectAttention(out_dims=out_dims, types=types, num_attn=1)
    inter_dec_off = BoxOffsetIntersection(dim=out_dims)
    # inter_attn = IntersectConcatAttention(query_dims=model_out_dims, key_dims=model_out_dims, num_attn=1, activation='leakyrelu', f_activation='sigmoid', layernorm=True, use_post_mat=True)

    # model
    enc_dec = QueryEncoderDecoder(graph=graph, enc=enc, path_dec=dec, inter_dec_cen=inter_dec_cen, inter_dec_off=inter_dec_off, use_inter_node=do_train)
    enc_dec.to('cuda')

    return enc_dec


def get_batch(queries, iteration, batch_size):
    # num_queries: a list of num of queries per formula
    num_queries = [float(len(queries)) for queries in queries.values()]

    # Use the num of queries per formula to form a multinomial dist to randomly pick on value
    formula_index = np.argmax(np.random.multinomial(1, np.array(num_queries) / float(sum(num_queries))))
    formula = list(queries.keys())[formula_index]

    # Total number of queries of this formula
    n = len(queries[formula])

    # Specify the window
    start = (iteration * batch_size) % n
    end = min(((iteration + 1) * batch_size) % n, n)
    end = n if end <= start else end

    queries = queries[formula][start:end]
    return formula, queries


def train(model, optimizer, batch_size, train_queries, val_queries, max_iter):
    inter_weight = 0.005
    path_weight = 0.01
    losses = []

    for iteration in range(max_iter):
        loss = 0

        # Reset gradients
        optimizer.zero_grad()

        for query_type in train_queries:
            # Get the bach / for each batch
            formula, train_batch = get_batch(train_queries[query_type], iteration, batch_size)

            if query_type == '1-chain':
                # Compute loss
                loss += model.box_loss(formula, train_batch)

            elif 'inter' in query_type:
                # Compute loss
                loss += inter_weight * model.box_loss(formula, train_batch)
                loss += inter_weight * model.box_loss(formula, train_batch, hard_negatives=True)

            else:
                # Compute loss
                loss += path_weight * model.box_loss(formula, train_batch)

        # Update loss
        losses.append(loss)

        # Compute Gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Validate
        aucs, aprs = test(model, val_queries, )


def test(model, queries, batch_size=1000):
    """
    Given queries, evaluate AUC and APR by negative sampling and hard negative sampling
    Args:
        queries:
            key: "full_neg" (full negative sample) or "one_neg" (only one negative sample)
            value: a dict()
                key: query type
                value: a dict()
                    key: formula template
                    value: the query object
    Return:
        aucs: a dict()
            key: query type, or query_type+"hard"
            value: AUC for this query type

    """

    aucs = {}
    aprs = {}
    random.seed(0)

    for query_type in queries['one_neg']:

        # Use Area Under ROC Curve (AUC) metric for current query type
        labels = []
        predictions = []
        for formula in queries["one_neg"][query_type]:
            formula_queries = queries["one_neg"][query_type][formula]
            formula_labels = []
            formula_predictions = []

            # split the formula_queries intp batches, add collect their ground truth and prediction scores
            offset = 0
            while offset < len(formula_queries):

                # Get the queries of the batch
                max_index = min(offset + batch_size, len(formula_queries))
                batch_queries = formula_queries[offset:max_index]

                # Get one random negative sample for batch size entities
                lengths = [1 for j in range(offset, max_index)]
                negatives = [random.choice(formula_queries[j].neg_samples) for j in range(offset, max_index)]

                offset += batch_size

                labels.extend([1 for _ in range(len(lengths))])
                formula_labels.extend([0 for _ in range(len(negatives))])

                # Get the scores of the batch
                batch_scores = model.forward(formula, batch_queries + [b for i, b in enumerate(batch_queries) for _ in range(lengths[i])], [q.target_node for q in batch_queries] + negatives)
                batch_scores = batch_scores.data.tolist()
                formula_predictions.extend(batch_scores)

            labels.extend(formula_labels)
            predictions.extend(formula_predictions)

        auc = roc_auc_score(labels, np.nan_to_num(predictions))

        # Use average percentiel rank (APR) metric for current query type
        perc_scores = []
        for formula in queries["full_neg"][query_type]:
            formula_queries = queries["full_neg"][query_type][formula]

            offset = 0
            while offset < len(formula_queries):

                # Get the queries of the batch
                max_index = min(offset + batch_size, len(formula_queries))
                batch_queries = formula_queries[offset:max_index]

                # Get all the negative samples for batch size entities
                lengths = [len(formula_queries[j].neg_samples) for j in range(offset, max_index)]
                negatives = [n for j in range(offset, max_index) for n in formula_queries[j].neg_samples]

                offset += batch_size

                batch_scores = model.forward(formula, batch_queries + [b for i, b in enumerate(batch_queries) for _ in range(lengths[i])], [q.target_node for q in batch_queries] + negatives)
                batch_scores = batch_scores.data.tolist()

                # Percentile rank score: Given a query, one positive target cos score p, x negative target, and their cos score [n1, n2, ..., nx], See the rank of p in [n1, n2, ..., nx]
                batch_perc_scores = []
                cum_sum = 0
                neg_scores = batch_scores[len(lengths)]
                for i, length in enumerate(lengths):
                    # score[i]: the cos score for positive query-target
                    # neg_scores[cum_sum:cum_sum+length]: the list of cos score for negative query-target
                    perc_scores.append(stats.percentileofscore(neg_scores[cum_sum:cum_sum + length], batch_scores[i]))
                    cum_sum += length
                perc_scores.extend(batch_perc_scores)

        perc = np.mean(perc_scores)

        aucs[query_type] = auc
        aprs[query_type] = perc

    return aucs, aprs
