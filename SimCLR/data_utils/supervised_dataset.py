from torchvision.transforms import transforms
from torchvision import transforms, datasets
from exceptions.exceptions import InvalidDatasetSelection


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
        normalize = transforms.Normalize(mean=mean, std=std)
        if mode == 'train':
            data_transforms = transforms.Compose([
                                                transforms.RandomCrop(size=size, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize,
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