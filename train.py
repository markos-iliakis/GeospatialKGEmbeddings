import torch.autograd
from torch import optim

from Yago2GeoDatasetHelpers.create_data_files import create_id2type
from Yago2GeoDatasetHelpers.dataset_helper_functions import read_graph, read_id2geo, read_id2extent, pickle_load
from Yago2GeoDatasetHelpers.query_sampling import load_queries_by_formula, load_test_queries_by_formula
from help_functions import create_paths, create_architecture, train, load_data

if __name__ == '__main__':
    batch_size = 2048
    feat_embed_dim = 128
    spa_embed_dim = 128
    max_iter = 22100
    lr = 0.001
    name = f'se-kge_iter-{max_iter-100}_feat-{feat_embed_dim}_spa-{spa_embed_dim}'

    # # create id2type
    # paths = create_paths()
    # create_id2type(paths['entitiesID_path'], paths['classes_path'], paths['data_path'] + 'id2type.json')

    # Load Data
    print('Loading graph data..')
    data = load_data(feat_embed_dim)

    # Create model
    print('Creating Architecture..')
    model = create_architecture(data['path'], data['graph'], data['feature_modules'], feat_embed_dim, spa_embed_dim, do_train=True)

    # Create optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr)

    # Train the model
    print('Training..')
    train(model, optimizer, batch_size, data, max_iter, name)
