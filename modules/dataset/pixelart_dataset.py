
import kagglehub
import numpy as np
import torch
from torch.utils.data import Dataset

CATEGORIES = [ 'human', 'non-human', 'food', 'spell', 'side-facing' ]

class PixelartDataset(Dataset):
    def __init__(self, transform, null_context=False):
        # Download latest version
        path = kagglehub.dataset_download("ebrahimelgazar/pixel-art")

        self.sprites = np.load(path+"/sprites.npy")
        self.slabels = np.load(path+"/sprites_labels.npy")
        print(f"sprite shape: {self.sprites.shape}")
        print(f"labels shape: {self.slabels.shape}")
        self.transform = transform
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape

    # Return the number of images in the dataset
    def __len__(self):
        return len(self.sprites)

    # Get the image and label at a given index
    def __getitem__(self, idx):
        # Return the image and label as a tuple
        if self.transform:
            image = self.transform(self.sprites[idx])
        else:
            image = self.sprites[idx]
        if self.null_context:
            label = torch.tensor(0).to(torch.int64)
        else:
            label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

    def getshapes(self):
        # return shapes of data and labels
        return self.sprites_shape, self.slabel_shape


from torch.utils.data import Dataset


class FilteredDataset(Dataset):
    def __init__(self, base_dataset, target_classes):
        """
        base_dataset: dataset original (ex: torchvision.datasets.CIFAR10)
        target_classes: int ou lista de ints com os índices das classes desejadas
        """
        if isinstance(target_classes, int):
            target_classes = [target_classes]

        self.base_dataset = base_dataset
        self.target_classes = target_classes

        # Filtrar índices das amostras com as classes desejadas
        self.indices = [
            i for i, (_, label) in enumerate(base_dataset) if label in target_classes
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.base_dataset[real_idx]


class FilteredDatasetOneHot(Dataset):
    def __init__(self, base_dataset, target_classes):
        """
        base_dataset: dataset original que devolve (image, one_hot_label)
        target_classes: int ou lista de ints com os índices das classes desejadas
        """
        if isinstance(target_classes, int):
            target_classes = [target_classes]

        self.base_dataset = base_dataset
        self.target_classes = target_classes

        # Converter target_classes em tensor booleano para facilitar comparação
        self.target_mask = torch.zeros(base_dataset[0][1].shape[0])
        self.target_mask[target_classes] = 1

        # Filtrar índices onde a classe é uma das desejadas
        self.indices = [
            i for i in range(len(base_dataset))
            if (torch.argmax(base_dataset[i][1]) in target_classes)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.base_dataset[real_idx]


def onehotmatrix2labels(one_hot_matrix):
    return [CATEGORIES[np.argmax(vec)] for vec in one_hot_matrix]



def onehot2label(one_hot):
    # return [CATEGORIES[np.argmax(vec)] for vec in one_hot_matrix]
    # Obter o índice do valor 1
    index = np.argmax(one_hot)

    # Obter a categoria correspondente
    return CATEGORIES[index]