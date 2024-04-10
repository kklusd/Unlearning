from SimCLR.models.resnet_classifier import ResNetClassifier
import torch.nn as nn
import numpy as np
import torch
from SimCLR.SupClassifier import SupClassifier
from dataset import RetrainData
import json
from torch.utils.data import DataLoader

def get_retrain_para(para_path):
    with open(para_path, 'r') as f:
        para_dict = json.load(f)
        f.close()
    return para_dict
def set_retrain_loader(data, args, mode="train"):
    if mode == "train":
        batch_size = args["batch_size"]
    else:
        batch_size = 100
    workers = args['workers']
    data_set = RetrainData(data)
    retrain_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True,num_workers=workers, pin_memory=True, sampler=None)
    return retrain_loader

def retrain(args, retrain_loader, val_loader):
    model = ResNetClassifier(base_model=args["arch"], num_class=args["out_dim"], weights='IMAGENET1K_V1')
    optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"], momentum=args["mo"], weight_decay=["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    with torch.cuda.device(args.gpu_index):
        classifier = SupClassifier(model=model, optimizer=optimizer, scheduler=scheduler, train_loader=retrain_loader, val_loader=val_loader, args=args)
        classifier.train()
    return model
