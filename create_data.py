from help_functions import create_data, create_paths

if __name__ == '__main__':
    # Create paths
    print('Creating Paths..')
    data_path, classes_path, entitiesID_path, graph_path, triples_path, types_geo_path, train_path, valid_path, test_path = create_paths()

    # Create Data
    print('Creating Data..')
    create_data(data_path, classes_path, entitiesID_path, graph_path, triples_path, types_geo_path)

    print('Done!')
