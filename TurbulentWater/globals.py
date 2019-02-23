import os

import torch

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


def cuda_if_available(x):
    if torch.cuda.is_available():
        x = x.cuda()

    return x


def map_location():
    if torch.cuda.is_available():
        return None

    return 'cpu'
