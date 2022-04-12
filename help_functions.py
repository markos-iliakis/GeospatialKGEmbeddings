import torch

from Model.MultiLayerFFN import MultiLayerFeedForwardNN
from Model.decoder import BilinearBlockDiagMetapathDecoder, SimpleSetIntersection, IntersectConcatAttention
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

    return data_path, classes_path, entitiesID_path, graph_path, triples_path, types_geo_path


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


def create_architecture(graph, feature_modules, model_out_dims, feat_dims, spa_embed_dim, do_train):
    print('Creating Encoder Operator..')
    # encoder
    feat_enc = DirectEncoder(graph.features, feature_modules)
    ffn = MultiLayerFeedForwardNN(input_dim=6 * 16, output_dim=64, num_hidden_layers=1, dropout_rate=0.5, hidden_dim=512, activation='sigmoid', use_layernormalize=True, skip_connection=True)
    spa_enc = TheoryGridCellSpatialRelationEncoder(spa_embed_dim=64, coord_dim=2, frequency_num=16, max_radius=5400000, min_radius=50, freq_init='geometric', ffn=ffn, device='cuda')
    pos_enc = ExtentPositionEncoder(id2geo=read_id2geo('./Data/id2geo.json'), id2extent=read_id2extent('./Data/id2geo.json'), spa_enc=spa_enc, graph=graph, device='cuda')
    enc = NodeEncoder(feat_enc, pos_enc, agg_type='concat')

    print('Creating Projection Operator..')
    # decoder-projection
    dec = BilinearBlockDiagMetapathDecoder(graph.relations, feat_dims=feat_dims, spa_embed_dim=spa_embed_dim)

    print('Creating Intersection Operator..')
    # intersection-attention
    inter_dec = SimpleSetIntersection(agg_func=torch.mean)
    inter_attn = IntersectConcatAttention(query_dims=model_out_dims, key_dims=model_out_dims, num_attn=1, activation='leakyrelu', f_activation='sigmoid', layernorm=True, use_post_mat=True)

    # model
    enc_dec = QueryEncoderDecoder(graph=graph, enc=enc, path_dec=dec, inter_dec=inter_dec, inter_attn=inter_attn, use_inter_node=do_train)
    enc_dec.to('cuda')

    return enc_dec


def train(model, optimizer, batch_size, epochs):

    for epoch in range(epochs):
        # Reset gradients
        optimizer.zero_grad()

        # Get the bach / for each batch

        # Compute loss
        loss = model.margin_loss()  # params: formula queries hard_negatives

        # Update loss

        # Compute Gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Validate

