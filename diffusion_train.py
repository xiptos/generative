# hyperparameters
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import torch.nn.functional as F

from modules.diffusion.context_unet import ContextUnet
from modules.diffusion.diffusion_utils import perturb_input
from modules.dataset.pixelart_dataset import PixelartDataset


def train(nn_model: nn.Module, dataloader: DataLoader, ab_t):
    # training with context code
    # set into train mode
    nn_model.train()

    for ep in range(n_epoch):
        print(f"epoch {ep}")

        # linearly decay learning rate
        optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)

        for x, c in dataloader:  # x: images  c: context
            optim.zero_grad()
            x = x.to(device)
            c = c.to(x)
            print(c)

            # randomly mask out c
            context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
            c = c * context_mask.unsqueeze(-1)
            print(c)

            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
            x_pert = perturb_input(ab_t, x, t, noise)

            # use network to recover noise
            pred_noise = nn_model(x_pert, t / timesteps, c=c)

            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            optim.step()

        # save model periodically
        if ep % 4 == 0 or ep == int(n_epoch - 1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(nn_model.state_dict(), save_dir + f"context_model_{ep}.pth")
            print("saved model at " + save_dir + f"context_model_{ep}.pth")

if __name__ == '__main__':

    image_size = 16

    # diffusion hyperparameters
    timesteps = 500
    beta1 = 1e-4
    beta2 = 0.02

    # network hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    n_feat = 64 # 64 hidden dimension feature
    n_cfeat = 5 # context vector is of size 5
    height = 16 # 16x16 image
    save_dir = './results/diffusion/weights/'

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )  # used when transforming image to tensor

    # training hyperparameters
    batch_size = 100
    n_epoch = 32
    lrate=1e-3

    # construct DDPM noise schedule
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    # construct model
    # nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
    # reset neural network
    nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

    # re setup optimizer
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = PixelartDataset(transform=transform)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True)

    train(nn_model, dataloader, ab_t)
