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
    
    
class ExpandDim(object):
    """
        Expand 1 channel image from shape [H, W] to [1, H, W]
    """
    def __init__(self):
        pass
    
    def __call__(self, x):
        return x[None, :, :]


class AddGaussianNoise(object):
    """
        Add random noise to input
    """
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        return x + (torch.randn(x.size())*self.std + self.mean)
    

class AddDropoutNoise(object):
    """
        Add random dropout noise to the 2D input, used in the original of denoising autoencoder
    """
    def __init__(self, p):
        assert p >= 0.0 and p <= 1.0, "[ERROR]: The value of p should be between 0 and 1!"
        self.p = p
    
    def __call__(self, x):
        # Each activation has a p% chance to be activated, very similar to dropout
        mask = torch.bernoulli(torch.ones(x.shape())*self.p)
        return x*mask