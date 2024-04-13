import copy
import os
import time
from collections import OrderedDict
import pickle
from mu.bad_teaching import set_dataset
import salUN.arg_parser_salun
import salUN.evaluation as evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import salUN.unlearn as unlearn
import salUN.utils as utils
from salUN.trainer import validate
import mu.arg_parser as parser
from torch.utils.data import DataLoader
from salUN.trainer import train, validate
from mu.mu_retrain import *
from salUN.utils import setup_seed
from salUN.generate_mask import save_gradient_ratio

def set_salUN_loader(forget_set,retain_set, opt):
    forget_train = copy.deepcopy(forget_set['train'])
    forget_val = forget_set['val']
    retain_train = retain_set['train']
    retain_val = retain_set['val']
    ret_loader = DataLoader(
        retain_train,
        batch_size=256,
        shuffle=True)
    for_loader = DataLoader(
        forget_train,
        batch_size=256,
        shuffle=True)
    vall_loader = DataLoader(
        retain_val,
        batch_size=256,
        shuffle=True)
    unlearn_data_loaders = OrderedDict(
        retain=ret_loader, forget=for_loader, test=vall_loader
    )
    return unlearn_data_loaders

def salUN_model_loader(opt, device):
    param_path = 'SimCLR/runs/params.json'
    params = get_retrain_para(param_path)
    raw_check_point = torch.load('./SimCLR/runs/original_model/checkpoint_0300.pth.tar', map_location='cuda:0')
    model = ResNetClassifier(params["arch"], num_class=opt.num_class, weights='IMAGENET1K_V1')
    model.load_state_dict(raw_check_point['state_dict'])
    model.to(device)
    model_dic = {'raw_model': model}
    return model_dic

def salUN_process(unlearn_data_loader,model,opt,forget_set,retain_set):
    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if opt.mask_path == None:
        save_gradient_ratio(unlearn_data_loader, model, criterion, opt)
    else:
        mask = torch.load(opt.mask_path)
        unlearn_method = salUN.unlearn.get_unlearn_method(opt.unlearn)
        unlearn_method(unlearn_data_loader, model, criterion, opt, mask)
        salUN.unlearn.save_unlearn_checkpoint(model, None, opt)

    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loader.items():
            # salUN.utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, opt)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        evaluation_result["accuracy"] = accuracy

    forget_train = copy.deepcopy(forget_set['train'])
    forget_val = forget_set['val']
    retain_train = retain_set['train']
    retain_val = retain_set['val']
    shadow_train = torch.utils.data.Subset(retain_train, list(range(2000)))
    shadow_train_loader = torch.utils.data.DataLoader(
        shadow_train, batch_size=opt.batch_size, shuffle=False
    )
    shadow_test = torch.utils.data.Subset(retain_val, list(range(1000)))
    shadow_test_loader = torch.utils.data.DataLoader(
        shadow_test, batch_size=opt.batch_size, shuffle=False
    )
    target_test = torch.utils.data.Subset(forget_train, list(range(500)))
    target_test_loader = torch.utils.data.DataLoader(
        target_test, batch_size=opt.batch_size, shuffle=False
    )

    salUN.evaluation.SVC_MIA(
        shadow_train=shadow_train_loader,
        shadow_test=shadow_test_loader,
        target_train=None,
        target_test=target_test_loader,
        model=model,
    )
    print(evaluation_result)
