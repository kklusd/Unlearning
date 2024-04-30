import json
from SimCLR.models.resnet_classifier import ResNetClassifier
import torch
from torchvision.models import resnet18, resnet34, resnet50, wide_resnet50_2
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
from torch_cka import CKA

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

batch_size = 256

dataset = CIFAR10(root='../datasets/',
                  train=False,
                  download=True,
                  transform=transform)
datasett = torch.utils.data.Subset(dataset, range(1000))
dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g,)


#===============================================================
model1 = resnet18(pretrained=True)


#
param_path = '../SimCLR/runs/params.json'
with open(param_path, 'r') as f:
    params = json.load(f)
    f.close()

raw_check_point = torch.load('../SimCLR/runs/original_model/checkpoint_0300.pth.tar', map_location='cuda:0')
model2 = ResNetClassifier(params["arch"], num_class=10, weights='IMAGENET1K_V1')
model2.load_state_dict(raw_check_point['state_dict'])



cka = CKA(model1, model2,
        model1_name="ResNet18", model2_name="ResNet18_pretrained",
        device='cuda')

cka.compare(dataloader)

cka.plot_results(save_path="assets/resnet-resnet_compare.png")