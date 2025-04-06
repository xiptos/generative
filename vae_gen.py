import torch
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from modules.dataset.pixelart_dataset import FilteredDatasetOneHot, PixelartDataset
from modules.vae.vae import PixelVAE
from vae_train import transform, IMAGE_SIZE, LATENT_DIM

MODEL_FILE = './results/vae/weights/vae_model_50.pth'

if __name__ == '__main__':
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    dataset = FilteredDatasetOneHot(PixelartDataset(transform=transform), target_classes=[4])
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    model = PixelVAE(image_size=IMAGE_SIZE, latend_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    for pic, _ in loader:  # batch size is 1, loader is shuffled, so this gets one random pic
        pics = pic.to(device)
        break
    orig = torch.clone(pics)

    for _ in range(1):
        recon, mu, log = model(pics)
        pic = recon[0].view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        pics = torch.cat((pics, pic), dim=0)

    save_image(pics, 'results/gan/results/rndpics.jpg', nrow=8)


    # use code below if you want to manually tweak the latent vector
    # mu, log_var = model.encode(orig)

    # for _ in range(7):
    #     w = 1e-11
    #     std = torch.exp(w * log_var)
    #     eps = torch.randn_like(std)
    #     z = eps * std + mu
    #     recon = model.decode(z)
    #     pic = recon[0].view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    #     pics = torch.cat((pics, pic), dim=0)

    # save_image(pics, 'rndpics.jpg', nrow=8)