from torchvision.transforms import v2
import numpy as np
import torch


class Flatten(object):
    """
        Flatten Image into 1D Tensor, use in the Linear MNIST AutoEncoder
    """
    def __init__(self):
        pass
    
    def __call__(self, x):
        return torch.flatten(x)
    
    
class Linear_Normalize(object):
    """
        Normalize for 1D data, use in the Linear MNIST AutoEncoder
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, x):
        return (x - self.mean)/self.std
