from torch.utils.data import Dataset
import torch
from network import Generator, Discriminator
import torch.nn as nn
torch.random.seed(123)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)   


class FeaturesSet(Dataset):
    def __init__(self, forget_features, retain_features):
        super().__init__()
        self.forget_features = forget_features
        self.retain_features = retain_features
        self.forget_set_len = self.forget_features.shape[0]
        self.retain_set_len = self.retain_features.shape[0]
    def __len__(self):
        return self.forget_set_len + self.retain_set_len
    
    def __getitem__(self, index):
        if (index < self.forget_set_len):
            x = self.forget_features[index]
            y = 1
            return x, y
        else:
            x = self.retain_features[index - self.forget_set_len]
            y = 0
            return x, y
        
    
def model_init(args, device):
    new_generator = Generator(nz=args.noise_size, ngf=args.ngf, nc=args.ots_size).to(device)
    new_generator.apply(weights_init)
    new_discriminator = Discriminator(nc=args.ots_size, ndf=args.ndf).to(device)
    new_discriminator.apply(weights_init)
    return new_generator, new_discriminator