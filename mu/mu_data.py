import torch

from ARPL.models import gan
import numpy as np

import random
def aug(dataset):

    len = 300
    device = torch.device('cpu')
    model_path = 'log/models/un/resnet_un_ARPLoss_0.1_True_G.pth'
    netG = gan._netG32(1,nz=100, ngf=64, nc=3)
    check_point = torch.load(model_path, map_location=torch.device('cpu'))
    netG.load_state_dict(check_point,False)
    netG = netG.to(device)
    noise = torch.FloatTensor(len, 100, 1, 1).normal_(0, 1)
    fake = netG(noise)
    for i in range(len):
        dataset.append((fake[i].detach(), random.randint(0,9)))
    return dataset