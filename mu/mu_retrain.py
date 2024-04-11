from SimCLR.models.resnet_classifier import ResNetClassifier
import torch.nn as nn
import numpy as np
import torch
from SimCLR.SupClassifier import SupClassifier
from .dataset import RetrainData
import json
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from SimCLR.data_utils.supervised_dataset import Cutout


def model_loader(opt, device, raw_model_path, retrain_model_path):
    raw_check_point = torch.load(raw_model_path, map_location=device)
    raw_model = ResNetClassifier(num_class=opt.num_class,base_model=opt.base_model, weights=None)
    raw_model.load_state_dict(raw_check_point['state_dict'])
    raw_model.to(device)
    retrain_check_point = torch.load(retrain_model_path, map_location=device)
    retrain_model = ResNetClassifier(num_class=opt.num_class,base_model=opt.base_model, weights=None)
    retrain_model.load_state_dict(retrain_check_point['state_dict'])
    raw_model.to(device)
    model_dic = {'raw_model': raw_model, 'retrain_model': retrain_model}
    return model_dic

def get_retrain_para(para_path):
    with open(para_path, 'r') as f:
        para_dict = json.load(f)
        f.close()
    return para_dict
def set_retrain_loader(root_folder, retain_indexes, args, mode="train"):
    workers = args['workers']
    if mode == "train":
        batch_size = args["batch_size"]
        data_transforms = transforms.Compose([
                                    transforms.RandomCrop(size=32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                    Cutout(n_holes=1, length=16),
                                    ])
        shuffle = True
    else:
        batch_size = 100
        data_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                    ])
        shuffle = False
    data_set = RetrainData(root_folder, retain_indexes, mode, transform=data_transforms)
    retrain_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True, sampler=None)
    return retrain_loader

def retrain(args, retrain_loader, val_loader):
    model = ResNetClassifier(base_model=args["arch"], num_class=args["out_dim"], weights='IMAGENET1K_V1')
    device = torch.device("cuda:{}".format(args['gpu_index']))
    classifier = SupClassifier(model=model, train_loader=retrain_loader, val_loader=val_loader, epochs=args["epochs"], lr=args["lr"], device=device)
    classifier.train()
    return model
