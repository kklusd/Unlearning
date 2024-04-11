from torchvision.transforms import transforms
from torchvision import transforms, datasets
from ..exceptions.exceptions import InvalidDatasetSelection
import numpy as np
import torch

class Cutout(object):
    """
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class SupervisedLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_supervised_pipeline_transform(name, mode='train'):
        if name == 'cifar10':
            size = 32
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif name == 'stl10':
            size = 96
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        else:
            raise ValueError(name)
        normalize = transforms.Normalize(mean=mean, std=std)
        if mode == 'train':
            data_transforms = transforms.Compose([
                                                transforms.RandomCrop(size=size, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize,
                                                Cutout(n_holes=1, length=16),
                                                ])
        else:
            data_transforms = transforms.Compose([
                                                transforms.ToTensor(),
                                                normalize,
                                                ])
        return data_transforms

    def get_dataset(self, name):
        print(self.root_folder)
        train_transform = self.get_supervised_pipeline_transform(name, 'train')
        val_transform = self.get_supervised_pipeline_transform(name, 'val')
        if name == 'cifar10':
            train_dataset = datasets.CIFAR10(self.root_folder,transform=train_transform,
                                                              download=True)
            val_dataset = datasets.CIFAR10(self.root_folder,train=False, transform=val_transform)
        elif name == 'stl10':
            train_dataset = datasets.STL10(self.root_folder,transform=train_transform,split='train',
                                                              download=True)
            val_dataset = datasets.STL10(self.root_folder,split='test', transform=val_transform)
        else:
            raise ValueError(name)
        return train_dataset, val_dataset