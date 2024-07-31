import torch
import torch.nn as nn


class BernoulliKLDiv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rho_hat, rho):
        rho*(torch.log(rho) - torch.log(rho_hat)) + (1 - rho)*(torch.log(1 - rho) - torch.log(1 - rho_hat))


class CustomSparseAutoencoderLoss(nn.Module):
    """
        Sparse Autoencoder Loss with custom Kulback-Leibler Divergence implementation
    """
    def __init__(self, beta, rho):
        super().__init__()
        self.beta = beta
        self.rho = rho
        self.mse = nn.MSELoss()
        self.KLDiv = BernoulliKLDiv()

    def forward(self, output, target, embedding):
        rho_hat = torch.mean(embedding, dim=-1)
        return self.mse(output, target) + self.beta*self.KLDiv(rho_hat, self.rho)
    

class SparseAutoencoderLoss(nn.Module):
    """
        Sparse Autoencoder Loss with standard Kulback-Leibler Divergence Loss
    """
    def __init__(self, beta, rho, device):
        super().__init__()
        self.beta = beta
        self.rho = rho
        self.device = device
        self.mse = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean", log_target=False)
        self.KLDiv2 = nn.KLDivLoss(reduction="batchmean", log_target=False)

    def forward(self, output, target, embedding):
        rho_hat = torch.mean(embedding, dim=-1)
        rho = torch.ones(rho_hat.shape) * self.rho
        rho = rho.to(self.device)
        return self.mse(output, target) + self.beta*(self.KLDiv(rho_hat.log(), rho) + self.KLDiv2((1 - rho_hat).log(), (1 - rho)))