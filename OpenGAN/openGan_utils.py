from torch.utils.data import Dataset
import torch
from .network import Generator, Discriminator
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

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

def t_sne_visial2(real_features, gen_features):
    real_fea_len = real_features.shape[0]
    gen_fea_len = gen_features.shape[0]
    features = np.concatenate([real_features, gen_features], axis=0)
    tsne_result = TSNE(n_components=2).fit_transform(features)
    def scale_to_01_range(x):
        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range
    tsne_x = scale_to_01_range(tsne_result[:,0])
    tsne_y = scale_to_01_range(tsne_result[:,1])
    colors = ['b', 'c']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(tsne_x[0:real_fea_len], tsne_y[0:real_fea_len], c=colors[0], label='real',s=2)
    ax.scatter(tsne_x[real_fea_len:], tsne_y[real_fea_len:], c=colors[1], label='generate',s=1)
    ax.legend(loc='best')
    plt.savefig("tsne.png")


def t_sne_visial(retain_features,forget_features, gen_features):
    retain_fea_len = retain_features.shape[0]
    forget_fea_len = forget_features.shape[0]
    gen_fea_len = gen_features.shape[0]
    features = np.concatenate([retain_features,forget_features, gen_features], axis=0)
    tsne_result = TSNE(n_components=3).fit_transform(features)
    def scale_to_01_range(x):
        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range
    tsne_x = scale_to_01_range(tsne_result[:,0])
    tsne_y = scale_to_01_range(tsne_result[:,1])
    colors = ['b', 'c','r']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(tsne_x[0:retain_fea_len], tsne_y[0:retain_fea_len], c=colors[0], label='retain',s=4)
    ax.scatter(tsne_x[retain_fea_len:retain_fea_len+forget_fea_len], tsne_y[retain_fea_len:retain_fea_len+forget_fea_len], c=colors[1], label='forget',s=2)
    ax.scatter(tsne_x[retain_fea_len+forget_fea_len:], tsne_y[retain_fea_len+forget_fea_len:], c=colors[2], label='generate',s=1)

    ax.legend(loc='best')
    plt.savefig("tsne.png")