import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, KMNIST
from torch.utils.data import Dataset
from pandas import DataFrame
import numpy as np
from PIL import Image
from ..utils import mkdir_if_missing

class MNISTRGB(MNIST):
    """MNIST Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class KMNISTRGB(KMNIST):
    """KMNIST Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNIST(object):
    def __init__(self, **options):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'mnist')

        pin_memory = True if options['use_gpu'] else False

        trainset = MNISTRGB(root=data_root, train=True, download=True, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = MNISTRGB(root=data_root, train=False, download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10

class KMNIST(object):
    def __init__(self, **options):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'kmnist')

        pin_memory = True if options['use_gpu'] else False

        trainset = KMNISTRGB(root=data_root, train=True, download=True, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = KMNISTRGB(root=data_root, train=False, download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10

class CIFAR10(object):
    def __init__(self, **options):

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'cifar10')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.num_classes = 10
        self.trainloader = trainloader
        self.testloader = testloader

class CIFAR100(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'cifar100')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.num_classes = 100
        self.trainloader = trainloader
        self.testloader = testloader


class SVHN(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'svhn')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.SVHN(root=data_root, split='train', download=True, transform=transform_train)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.SVHN(root=data_root, split='test', download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.num_classes = 10
        self.trainloader = trainloader
        self.testloader = testloader

#---NEW--------------------------------------
class UnlearningData(Dataset):
    def __init__(self, forget_data):
        super().__init__()
        self.forget_data = forget_data
        self.forget_len = len(forget_data)

    def __len__(self):
        return self.forget_len

    def __getitem__(self, index):
            x = self.forget_data[index][0]
            y = self.forget_data[index][1]
            return x,y

class UnCIFAR10(object):
    def __init__(self, forget_indexes,**options):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'cifar10')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)

        #-----------------------seperateintoDfDr-----------------------------

        forgetset = {'train': [], 'val': []}
        retainset = {'train': [], 'val': []}
        forget_indexes = forget_indexes
        retain_indexes = np.delete(np.linspace(0,49999,50000,dtype=int), forget_indexes)#为啥要-1???????????????
        for index in forget_indexes:
            forgetset['train'].append(trainset[index])

        for index in retain_indexes:
            retainset['train'].append(trainset[index])


        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        retainloader = torch.utils.data.DataLoader(
            UnlearningData(retainset), batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        forgetloader = torch.utils.data.DataLoader(
            UnlearningData(forgetset), batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.num_classes = 10
        self.trainloader = trainloader
        self.testloader = testloader
        self.forgetloader = forgetloader
        self.retainloader = retainloader

__factory = {
    'mnist': MNIST,
    'kmnist': KMNIST,
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'svhn':SVHN,
    'un':UnCIFAR10
}

def create(name, **options):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](**options)



