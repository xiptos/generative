#  based on https://github.com/pytorch/examples/blob/main/vae/main.py
import os
import torch
import torch.utils.data
from os import mkdir
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split

from modules.dataset.pixelart_dataset import FilteredDatasetOneHot, PixelartDataset
from modules.vae.vae import PixelVAE
from modules.vae.vae_model_utils import rndstr

EPOCHS = 50  # number of training epochs
BATCH_SIZE = 128  # for data loaders
IMAGE_SIZE = 64
LATENT_DIM = 128
image_dim = 3 * IMAGE_SIZE * IMAGE_SIZE  # 67500

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # range [-1,1]
])  # used when transforming image to tensor

transform1 = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, antialias=True),
    transforms.CenterCrop(IMAGE_SIZE)])  # used by decode method to transform final output

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print('EPOCHS', EPOCHS, 'BATCH_SIZE', BATCH_SIZE, 'device', device)

# for model and results
save_dir = './results/vae/'
os.makedirs(save_dir+"/weights", exist_ok=True)
os.makedirs(save_dir+"/results", exist_ok=True)

# download dataset
dataset = FilteredDatasetOneHot(PixelartDataset(transform=transform), target_classes=[4])
# Definir tamanhos (por exemplo, 80% treino, 20% teste)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Fazer a divisão
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# create train and test dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = PixelVAE(image_size=IMAGE_SIZE, latend_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, log_var):
    MSE =F.mse_loss(recon_x, x.view(-1, image_dim))
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    kld_weight = 0.00025
    loss = MSE + kld_weight * KLD
    return loss


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # torch.cuda.empty_cache()
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        log_var = torch.clamp_(log_var, -10, 10)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            test_loss += loss_function(recon_batch, data, mu, log_var).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)[:n]])
                save_image(comparison.cpu(),
                           f'{save_dir}/results/reconstruction_{str(epoch)}.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    print(f'epochs: {EPOCHS}')

    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        torch.save(model.state_dict(), f'{save_dir}/weights/vae_model_{epoch}.pth')
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, LATENT_DIM).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, IMAGE_SIZE, IMAGE_SIZE),
                       f'{save_dir}/results/sample_{str(epoch)}.png')