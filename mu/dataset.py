from torchvision.datasets import CIFAR10
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from SimCLR.data_utils.gaussian_blur import GaussianBlur

class ContrastiveViewGenerator:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def generate(self, x):
        img_ls = [x]
        for i in range(self.n_views - 1):
            img_ls.append(self.base_transform(x))
        return img_ls

class RetrainData(Dataset):
    def __init__(self, root_folder, retain_indexes, mode, transform=None):
        super().__init__()
        self.data = self.get_data(root_folder, retain_indexes, mode)
        self.data_len = len(self.data)
        self.transform = transform
    
    @staticmethod
    def get_data(root_folder, retain_indexes, mode):
        data_set = CIFAR10(root_folder, transform=None, download=True, train=mode=="train")
        data = []
        if mode == "train":
            for index in retain_indexes:
                data.append(data_set[index])
        else:
            for i in range(len(data_set)):
                data.append(data_set[i])
        return data

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        img = self.data[index][0]
        if self.transform is not None:
            img = self.transform(img)
        label = self.data[index][1]
        return img, label




class UnlearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)
    
    def __len__(self):
        return self.retain_len +self.forget_len
    
    def __getitem__(self, index):
        if (index < self.forget_len):
            x = self.forget_data[index][0]
            y = 1
            return x, y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x, y
        
class ContrastiveUnlearningData(Dataset):
    def __init__(self, forget_data, retain_data, data_name, n_views):
        super().__init__()
        self.data_name = data_name
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)
        self.n_views = n_views
        if data_name == 'cifar10':
            size = 32
            self.transform = self.get_simclr_pipeline_transform(size=size)
        elif data_name == 'stl10':
            size = 96
            self.transform = self.get_simclr_pipeline_transform(size=size)
        else:
            raise ValueError(data_name)
        self.view_generator = ContrastiveViewGenerator(self.transform, self.n_views)

    def __len__(self):
        return self.retain_len + self.forget_len
    
    def get_simclr_pipeline_transform(self, size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.ToPILImage(),
                                            transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * size)),
                                            transforms.ToTensor()])
        return data_transforms  
    
    def __getitem__(self, index):
        if (index < self.forget_len):
            x = self.forget_data[index][0]
            augmented_x =  self.view_generator.generate(x)
            y = 1
            return augmented_x, y
        else:
            x = self.retain_data[index - self.forget_len][0]
            augmented_x =  self.view_generator.generate(x)
            y = 0
            return augmented_x,  y


class BasicUnlearningData(Dataset):
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
