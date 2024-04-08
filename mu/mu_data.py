import torch
from torchvision.transforms import transforms
from ARPL.models import gan
import numpy as np
from SimCLR.data_utils.gaussian_blur import GaussianBlur

import random
def alpr_aug(dataset, aug_num):
    device = torch.device('cpu')
    model_path = 'log/models/un/resnet_un_ARPLoss_0.1_True_G.pth'
    netG = gan._netG32(1,nz=100, ngf=64, nc=3)
    check_point = torch.load(model_path, map_location=torch.device('cpu'))
    netG.load_state_dict(check_point,False)
    netG = netG.to(device)
    noise = torch.FloatTensor(len, 100, 1, 1).normal_(0, 1)
    fake = netG(noise)
    for i in range(aug_num):
        dataset.append((fake[i].detach(), random.randint(0,9)))
    return dataset

def simple_aug(dataset, aug_num, size=32):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    data_transforms = transforms.Compose([transforms.ToPILImage(),
                                            transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * size)),
                                            transforms.ToTensor()])
    dataset_len = len(dataset)
    for i in range(aug_num):
        if i >= dataset_len:
            index = dataset_len % i
            img = dataset[index][0]
            label = dataset[index][1]
            augment_img = data_transforms(img)
            dataset.append((augment_img, label))
        else:
            img = dataset[i][0]
            label = dataset[i][1]
            augment_img = data_transforms(img)
            dataset.append((augment_img, label))
    return dataset

