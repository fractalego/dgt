import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def set_global_device(string):
    device = torch.device(string)
