# %matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.utils as vutils

from modules.gan.gan import Generator

# custom weights initialization called on ``netG`` and ``netD``

if __name__ == '__main__':

    save_dir = "./results/gan/"

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 5

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 0

    # Decide which device we want to run on
    device = "cpu"
    if torch.has_mps:
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"

    # Create the generator
    netG = Generator(ngpu, nz, ngf, nc).to(device)
    netG.load_state_dict(torch.load(save_dir+"/weights/generator.pth"))
    netG.eval()

    # Print the model
    print(netG)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    img_list = []
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    with torch.no_grad():
        for _ in range(64):
            fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    # Plot the fake images from the last epoch
    # plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()
