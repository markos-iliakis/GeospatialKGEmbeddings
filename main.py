from torch import optim

from Yago2GeoDatasetHelpers.dataset_helper_functions import read_graph
from help_functions import create_paths, create_architecture, train, create_data

if __name__ == '__main__':
    batch_size = 2048
    feat_embed_dim = 64
    spa_embed_dim = 64
    epochs = 20
    data_path, classes_path, entitiesID_path, graph_path, triples_path, types_geo_path = create_paths()

    # Create Data
    print('Creating Data..')
    create_data(data_path, classes_path, entitiesID_path, graph_path, triples_path, types_geo_path)

    print('Loading graph data..')
    graph, feature_modules, node_maps = read_graph(graph_path)
    feat_dims = {e_type: feat_embed_dim for e_type in graph.relations}
    model_out_dims = {e_type: feat_dims[e_type] + spa_embed_dim for e_type in feat_dims}

    # Create model
    print('Creating Architecture..')
    model = create_architecture(graph, feature_modules, model_out_dims, feat_dims, spa_embed_dim, do_train=True)

    # Create optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

    # Train the model
    train(model, optimizer, batch_size)
