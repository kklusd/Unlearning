from torchvision.transforms import transforms
from torchvision import transforms, datasets
from exceptions.exceptions import InvalidDatasetSelection


class SupervisedLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_supervised_pipeline_transform(name, size, mode='train'):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        if name == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif name == 'stl10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        normalize = transforms.Normalize(mean=mean, std=std)
        if mode == 'train':
            data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(0.2,1)),
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
        train_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=self.get_simclr_pipeline_transform('cifar10', 32),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='train',
                                                          transform=self.get_simclr_pipeline_transform('stl10', '96'),
                                                          download=True)}

        val_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=False,
                                                        transform=self.get_simclr_pipeline_transform('cifar10', 32 ,'val'),
                                                        download=True),

                    'stl10': lambda: datasets.STL10(self.root_folder, split='test',
                                                    transform=self.get_simclr_pipeline_transform('stl10', '96', 'val'),
                                                    download=True)}
        try:
            train_dataset, val_dataset = train_datasets[name], val_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return train_dataset, val_dataset