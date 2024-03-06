from torchvision.datasets import CIFAR10
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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