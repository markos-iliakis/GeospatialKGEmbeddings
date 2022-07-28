import torch.autograd
from torch import optim

from Yago2GeoDatasetHelpers.dataset_helper_functions import read_graph
from Yago2GeoDatasetHelpers.query_sampling import load_queries_by_formula, load_test_queries_by_formula
from help_functions import create_paths, create_architecture, train, load_data

if __name__ == '__main__':
    batch_size = 2048
    feat_embed_dim = 128
    spa_embed_dim = 128
    max_iter = 22100
    lr = 0.01

    # Load Data
    print('Loading graph data..')
    data = load_data(feat_embed_dim)

    # Create model
    print('Creating Architecture..')
    model = create_architecture(data['path'], data['graph'], data['feature_modules'], feat_embed_dim, spa_embed_dim, do_train=True)

    # Create optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr)

    # Train the model
    torch.autograd.set_detect_anomaly(True)
    print('Training..')
    train(model, optimizer, batch_size, data['train_queries'], data['valid_queries'], max_iter)
