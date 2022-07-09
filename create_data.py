from help_functions import create_data, create_paths

if __name__ == '__main__':
    # Create paths
    print('Creating Paths..')
    paths = create_paths()

    # Create Data
    print('Creating Data..')
    create_data(paths)

    print('Done!')
