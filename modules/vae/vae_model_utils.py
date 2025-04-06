import random
from string import ascii_lowercase

import torch
from torch import nn


def loss_function(VAELossParams, kld_weight):
    recons, input, mu, log_var = VAELossParams
    recons_loss = nn.MSELoss(recons, input)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    )
    loss = recons_loss + kld_weight * kld_loss
    return {
        "loss": loss,
        "Reconstruction_Loss": recons_loss.detach(),
        "KLD": -kld_loss.detach(),
    }

def rndstr(n=6):
    return ''.join(random.choices(ascii_lowercase, k=n))
