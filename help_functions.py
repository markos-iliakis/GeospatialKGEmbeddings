import json
import os

import numpy as np
import random
import torch
from scipy.stats import stats
from sklearn.metrics import roc_auc_score
from statistics import mean

from Model.MultiLayerFFN import MultiLayerFeedForwardNN
from Model.decoder import BilinearBlockDiagMetapathDecoder, SimpleSetIntersection, IntersectConcatAttention, BoxDecoder, \
    BoxOffsetIntersection, CenterIntersection, BoxCenterIntersectAttention
from Model.encoder import DirectEncoder, TheoryGridCellSpatialRelationEncoder, ExtentPositionEncoder, NodeEncoder
from Model.model import QueryEncoderDecoder
from Yago2GeoDatasetHelpers.create_data_files import unite_files, make_id_files, make_custom_triples, \
    custom_triples_split, make_graph
from Yago2GeoDatasetHelpers.dataset_helper_functions import read_id2geo, read_id2extent, read_graph, pickle_load
from Yago2GeoDatasetHelpers.query_sampling import make_single_edge_query_data, sample_new_clean, \
    make_multiedge_query_data, make_inter_query_data, load_queries_by_formula, load_test_queries_by_formula


def create_paths():
    dataset = 'se-kge'  # yago2geo, se-kge
    sub_dataset = ''
    materialization = ''  # 'materialized/'
    triples_path = 'Datasets/' + dataset + sub_dataset + '/triples/' + materialization
    types_geo_path = 'Datasets/' + dataset + sub_dataset + '/geo_classes/'

    data_path = 'Data/' + dataset + sub_dataset + '/'
    classes_path = data_path + 'entity2type.json'
    entitiesID_path = data_path + 'entity2id.txt'
    graph_path = data_path + 'graph.pkl'
    train_path = data_path + 'train_queries/'
    valid_path = data_path + 'val_queries/'
    test_path = data_path + 'test_queries/'

    print(f'Training on {dataset + sub_dataset}')

    return {
        'data_path': data_path,
        'classes_path': classes_path,
        'entitiesID_path': entitiesID_path,
        'graph_path': graph_path,
        'triples_path': triples_path,
        'types_geo_path': types_geo_path,
        'train_path': train_path,
        'valid_path': valid_path,
        'test_path': test_path
    }


def create_data(paths):
    new_triples_path = paths['data_path'] + 'united_triples.nt'
    rid2inverse_path = paths['data_path'] + 'rid2inverse.json'
    relationsID_path = paths['data_path'] + 'relation2id.txt'
    id2geo_path = paths['data_path'] + 'id2geo.json'
    custom_triples = paths['data_path'] + 'custom_triples.txt'
    id2type_path = paths['data_path'] + 'id2type.json'

    # Unite all triples
    unite_files( paths['triples_path'],  paths['types_geo_path'], paths['data_path'])

    # Make the entity2id, relation2id with inverses, relations2inverse and relationsID2Inverse files
    make_id_files(new_triples_path,  paths['types_geo_path'], paths['data_path'])

    # Make the triples in the form of (subject_id, (subject_type, relation, object_type), object_id)
    custom_triples = make_custom_triples(new_triples_path,  paths['classes_path'], relationsID_path,  paths['entitiesID_path'], paths['data_path'])

    # Split the triples to train / valid / test and put them to pkl files
    custom_triples_split(custom_triples, paths['data_path'])

    # Make the graph file
    make_graph(custom_triples,  paths['classes_path'],  paths['entitiesID_path'],  paths['graph_path'], rid2inverse_path, id2type_path)

    # Make train / valid / test 1-chain queries
    make_single_edge_query_data(paths['data_path'],  paths['graph_path'], 100)  # 100

    # Make valid/testing 2/3 edges geographic queries, negative samples are geo-entities
    id2geo = read_id2geo(id2geo_path)
    sample_new_clean(paths['data_path'],  paths['graph_path'], id2geo=id2geo)

    # Make train x-inter queries, negative samples are geo-entities
    print("Do geo content sample")
    mp_result_geo_dir = paths['data_path'] + "train_inter_queries_geo_mp/"
    id2geo = read_id2geo(id2geo_path)
    make_inter_query_data(paths['data_path'],  paths['graph_path'], 50, 10000, max_inter_size=3, mp_result_dir=mp_result_geo_dir, id2geo=id2geo)  # 50 10000

    mp_result_geo_dir = paths['data_path'] + "train_queries_geo_mp/"
    id2geo = read_id2geo(id2geo_path)
    make_multiedge_query_data(paths['data_path'],  paths['graph_path'], 50, 20000, mp_result_dir=mp_result_geo_dir, id2geo=id2geo)  # 50 20000


def load_data(feat_embed_dim):
    data = dict()

    # Create paths
    print('Creating Paths..')
    paths = create_paths()
    data['path'] = paths['data_path']

    # Load Graph
    print('Loading Graph..')
    data['graph'], data['feature_modules'], data['node_maps'] = read_graph(paths['graph_path'], feat_embed_dim)

    # Load geo ids
    geo_info = pickle_load(data['path'] + 'id2geo_proj.pkl')
    geo_info_ext = pickle_load(data['path'] + 'id2extent_proj.pkl')
    data['geo_ids'] = list(geo_info.keys()) + list(geo_info_ext.keys())

    # Remove non-geo-ids ids from graph.full_lists
    removed_counter = 0
    for type in data['graph'].full_lists.keys():
        ids = data['graph'].full_lists[type].copy()
        for id in ids:
            if id not in data['geo_ids']:
                data['graph'].full_lists[type].remove(id)
                removed_counter += 1

    print(f'Removed {removed_counter} node ids from graph.full_lists')

    # Load queries of all types
    print('Loading Queries..')
    data['train_queries'] = dict()
    data['valid_queries'] = {'full_neg': dict(), 'one_neg': dict()}
    data['test_queries'] = {'full_neg': dict(), 'one_neg': dict()}

    removed_counter = 0
    for file in os.listdir(paths['train_path']):
        print(f'\t{file}')
        train_queries = load_queries_by_formula(paths['train_path'] + file)

        # check if every id is in geo ids
        for type in train_queries:
            for formula in train_queries[type]:
                queries = train_queries[type][formula].copy()
                for query in queries:
                    if invalid_ids(query, data['geo_ids']):
                        train_queries[type][formula].remove(query)
                        removed_counter += 1

        data['train_queries'].update(train_queries)

    print(f'Removed {removed_counter} train queries')

    removed_counter = 0
    for file in os.listdir(paths['valid_path']):
        print(f'\t{file}')
        valid_queries = load_test_queries_by_formula(paths['valid_path'] + file)

        # check if every id is in geo ids
        for neg_type in valid_queries:
            for type in valid_queries[neg_type]:
                for formula in valid_queries[neg_type][type]:
                    queries = valid_queries[neg_type][type][formula].copy()
                    for query in queries:
                        if invalid_ids(query, data['geo_ids']):
                            valid_queries[neg_type][type][formula].remove(query)
                            removed_counter += 1

        data['valid_queries']['full_neg'].update(valid_queries['full_neg'])
        data['valid_queries']['one_neg'].update(valid_queries['one_neg'])

    print(f'Removed {removed_counter} validation queries')

    # for file in os.listdir(test_path):
    #     print(f'\t{file}')
    #     x = load_test_queries_by_formula(test_path + file)
    #     data['test_queries']['full_neg'].update(x['full_neg'])
    #     data['test_queries']['one_neg'].update(x['one_neg'])

    return data


def invalid_ids(query, geo_ids):
    # Check query ids
    query_ids = query.anchor_nodes + [query.target_node]

    if query.hard_neg_samples is not None:
        query_ids += query.hard_neg_samples
    if query.neg_samples is not None:
        query_ids += query.neg_samples

    if not all(int(id) in geo_ids for id in query_ids):
        return True
    else:
        return False


def create_architecture(data_path, graph, feature_modules, feat_embed_dim, spa_embed_dim, do_train):
    out_dims = feat_embed_dim + spa_embed_dim
    types = [type for type in graph.relations]

    if 'se-kge' in data_path:
        geo_info = pickle_load(data_path + 'id2geo_proj.pkl')
        geo_info = {str(key): geo_info[key] for key in geo_info.keys()}
        geo_info_ext = pickle_load(data_path + 'id2extent_proj.pkl')
        geo_info_ext = {str(key): [[geo_info_ext[key][0], geo_info_ext[key][1]], [geo_info_ext[key][2], geo_info_ext[key][3]]] for key in geo_info_ext.keys()}
    else:
        geo_info = read_id2geo(data_path + 'id2geo.json')
        geo_info_ext = read_id2extent(data_path + 'id2geo.json')

    print('Creating Encoder Operator..')
    # encoder
    feat_enc = DirectEncoder(graph.features, feature_modules)
    ffn = MultiLayerFeedForwardNN(input_dim=6 * 16, output_dim=spa_embed_dim, num_hidden_layers=1, dropout_rate=0.5, hidden_dim=512, use_layernormalize=True, skip_connection=True)
    spa_enc = TheoryGridCellSpatialRelationEncoder(spa_embed_dim=spa_embed_dim, coord_dim=2, frequency_num=16, max_radius=5400000, min_radius=50, freq_init='geometric', ffn=ffn)
    pos_enc = ExtentPositionEncoder(id2geo=geo_info, id2extent=geo_info_ext, spa_enc=spa_enc, graph=graph)
    enc = NodeEncoder(feat_enc, pos_enc, agg_type='concat')

    print('Creating Projection Operator..')
    # decoder-projection
    dec = BoxDecoder(graph.relations, feat_embed_dim=feat_embed_dim, spa_embed_dim=spa_embed_dim)

    print('Creating Intersection Operator..')
    # intersection-attention
    inter_dec_cen = BoxCenterIntersectAttention(out_dims=out_dims, types=types, num_attn=1)
    inter_dec_off = BoxOffsetIntersection(dim=out_dims)

    # model
    enc_dec = QueryEncoderDecoder(graph=graph, enc=enc, path_dec=dec, inter_dec_cen=inter_dec_cen, inter_dec_off=inter_dec_off)
    enc_dec.to('cuda')

    return enc_dec


def check_conv(vals, tol):
    """
    Check the convergence of mode based on the evaluation score:
    Args:
        vals: a list of evaluation score
        tol: the threshold for convergence
    """
    if len(vals) < 4:
        return False
    last_vals = [x.data.cpu() for x in vals[-2:]]
    prev_vals = [x.data.cpu() for x in vals[-4:-2]]
    conv = np.mean(last_vals) - np.mean(prev_vals)
    return conv < tol


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


def train(model, optimizer, batch_size, data, max_iter, name):
    inter_weight = 0.005
    path_weight = 0.01
    losses = []

    for iteration in range(max_iter):
        model.train()
        loss = 0

        # Reset gradients
        optimizer.zero_grad()

        for query_type in data['train_queries']:
            # Get the bach / for each batch
            formula, train_batch = get_batch(data['train_queries'][query_type], iteration, batch_size)

            if query_type == '1-chain':
                # Compute loss
                loss += model.box_loss(formula, train_batch, data['geo_ids'])

            elif 'inter' in query_type:  # 2-inter, 3-inter, 3-inter-chain, 3-chain-inter
                # Compute loss
                loss += inter_weight * model.box_loss(formula, train_batch)
                loss += inter_weight * model.box_loss(formula, train_batch, hard_negatives=True)

            else:  # 2-chain, 3-chain
                # Compute loss
                loss += path_weight * model.box_loss(formula, train_batch)

        # Update losses
        losses.append(loss.data.tolist())

        # Compute Gradients
        loss.backward()

        # Update weights
        optimizer.step()

        if iteration % 1000 == 0:
            print('Validating..')
            # Validate
            aucs, aprs = test(model, data['valid_queries'])
            print(f'Iteration {iteration} : \n\taucs : \n\t\t{aucs} \n\taprs : \n\t\t{aprs} \n\tloss : \n\t\t{mean(losses)}')

            # Save the model
            torch.save(model.state_dict(), f'Checkpoints/{name}')

            print('Training..')


def test(model, queries, batch_size=2048):
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
    model.eval()

    # Get all the query types available
    query_types = [qt for qt in queries['one_neg']]

    for query_type in query_types:

        # Use Area Under ROC Curve (AUC) metric for current query type
        labels = []
        predictions = []
        for formula in queries["one_neg"][query_type]:
            formula_queries = queries["one_neg"][query_type][formula]
            formula_labels = []
            formula_predictions = []

            # split the formula_queries into batches, add collect their ground truth and prediction scores
            offset = 0
            while offset < len(formula_queries):

                # Get the queries of the batch
                max_index = min(offset + batch_size, len(formula_queries))
                batch_queries = formula_queries[offset:max_index]

                # Get one random negative sample for batch size entities
                lengths = [1 for j in range(offset, max_index)]
                negatives = [random.choice(formula_queries[j].neg_samples) for j in range(offset, max_index)]

                offset += batch_size

                formula_labels.extend([1 for _ in range(len(lengths))])
                formula_labels.extend([0 for _ in range(len(negatives))])

                # Get the scores of the batch
                batch_scores, in_box = model.forward(formula, batch_queries + [b for i, b in enumerate(batch_queries) for _ in range(lengths[i])], [q.target_node for q in batch_queries] + negatives)
                batch_scores = batch_scores.data.tolist()
                formula_predictions.extend(in_box)

            labels.extend(formula_labels)
            predictions.extend(formula_predictions)

        auc = roc_auc_score(labels, np.nan_to_num(predictions))

        # Use average percentile rank (APR) metric for current query type
        perc_scores = []
        for formula in queries["full_neg"][query_type]:
            formula_queries = queries["full_neg"][query_type][formula]

            offset = 0
            while offset < len(formula_queries):

                # Get the queries of the batch
                max_index = min(offset + batch_size, len(formula_queries))
                batch_queries = formula_queries[offset:max_index]

                # Get all the negative samples for batch size entities
                lengths = [len(formula_queries[j].neg_samples) for j in range(offset, max_index)]  # a list of N int, each indicate the negative sample size for this query
                negatives = [n for j in range(offset, max_index) for n in formula_queries[j].neg_samples]

                offset += batch_size

                # batch_scores : We have N queries. 1st N scores in batch_scores correspond to cos score for each positive query-target
                #                batch_scores[N:] correspond to cos score for each negative query-target which append in order, the total number of scores is sum(lengths)
                batch_scores, in_box = model.forward(formula, batch_queries + [b for i, b in enumerate(batch_queries) for _ in range(lengths[i])], [q.target_node for q in batch_queries] + negatives)
                batch_scores = batch_scores.data.tolist()

                # invert distances
                batch_scores = [1 / x for x in batch_scores]

                # Percentile rank score: Given a query, one positive target cos score p, x negative target, and their cos score [n1, n2, ..., nx], See the rank of p in [n1, n2, ..., nx]
                batch_perc_scores = []  # a list of percentile rank scores per query, APR is the average of all these scores
                cum_sum = 0
                neg_scores = batch_scores[len(lengths):]
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
