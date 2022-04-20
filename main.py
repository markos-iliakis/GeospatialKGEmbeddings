from torch import optim

from Yago2GeoDatasetHelpers.dataset_helper_functions import read_graph
from Yago2GeoDatasetHelpers.query_sampling import load_queries_by_formula, load_test_queries_by_formula
from help_functions import create_paths, create_architecture, train, create_data

if __name__ == '__main__':
    batch_size = 2048
    feat_embed_dim = 64
    spa_embed_dim = 64
    max_iter = 20
    lr = 0.01
    data_path, classes_path, entitiesID_path, graph_path, triples_path, types_geo_path, train_path, valid_path, test_path = create_paths()

    # Create Data
    print('Creating Data..')
    create_data(data_path, classes_path, entitiesID_path, graph_path, triples_path, types_geo_path)

    print('Loading graph data..')
    graph, feature_modules, node_maps = read_graph(graph_path)
    train_queries = load_queries_by_formula(train_path)
    valid_queries = load_test_queries_by_formula(valid_path)
    test_queries = load_test_queries_by_formula(test_path)

    # Create model
    print('Creating Architecture..')
    model = create_architecture(graph, feature_modules, feat_embed_dim, spa_embed_dim, do_train=True)

    # Create optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr)

    # Train the model
    train(model, optimizer, batch_size, train_queries, valid_queries, max_iter)
