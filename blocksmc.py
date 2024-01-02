import torch
from torch.distributions import Poisson, Normal, Uniform, Distribution, Categorical
from distributions import TruncatedDiagonalMVN
import numpy as np

# Configure GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

class BlockSMCSampler(object):
    def __init__(self):