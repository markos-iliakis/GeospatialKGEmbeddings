import torch

from help_functions import create_architecture

if __name__ == '__main__':
    model = create_architecture()
    model.load_state_dict(torch.load(PATH))
    model.eval()