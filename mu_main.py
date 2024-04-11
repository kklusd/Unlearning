import numpy as np
import torch
from mu.bad_teaching import set_dataset, set_loader, bad_teaching, bad_te_model_loader
from mu.mu_utils import Evaluation
import mu.arg_parser as parser
from mu.Scrub import scrub, scrub_model_loader
from mu.mu_basic import Neggrad,basic_model_loader,set_basic_loader,Retrain
import os
import pickle
from mu.mu_retrain import *
from mu.mu_data import alpr_aug, simple_aug
import copy
import time
def main():
    time1 = time.time()
    opt = parser.parse_option()
    method = opt.method
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoches = opt.epoches
    if opt.mode == 'random' and opt.saved_data_path != '':
        forget_data_file = os.path.join(opt.saved_data_path, 'forget_data.pt')
        retain_data_file = os.path.join(opt.saved_data_path, 'retain_data.pt')
        retain_indexes_file = os.path.join(opt.saved_data_path, 'retain_indexes.pt')
        with open(forget_data_file, 'rb') as f:
                forget_set = pickle.load(f)
                f.close()
        with open(retain_data_file, 'rb') as f:
            retain_set = pickle.load(f)
            f.close()
        with open(retain_indexes_file, 'rb') as f:
            retain_indexes = pickle.load(f)
            f.close()
    else:
        forget_set, retain_set, retain_indexes = set_dataset(opt.data_name, opt.data_root, mode=opt.mode,
                                            forget_classes=opt.forget_class, forget_num=opt.forget_num)
    forget_train = copy.deepcopy(forget_set['train'])
    forget_val = forget_set['val']
    retain_train = retain_set['train']
    retain_val = retain_set['val']
    if opt.data_augment == 'arpl':
        forget_train = alpr_aug(forget_train, opt.augment_num)
    elif opt.data_augment == 'simple':
        forget_train = simple_aug(forget_train, opt.augment_num)
#---------------------------------method-------------------------------------------
    if method == 'bad_teaching':
        model_dic = bad_te_model_loader(opt, device)
        # ------------------------------dataloader--------------------------------------------------
        unlearn_dl = set_loader(retain_train, forget_train, opt)

        # ----------------------------Training Process--------------------------------
        for i in range(epoches):
            epoch = i + 1
            bad_teaching(model_dic=model_dic, unlearing_loader=unlearn_dl, epoch=epoch, device=device, opt=opt)

        # ----------------------------Eva--------------------------------
        Evaluation(model_dic,retain_train, retain_val,forget_set['train'], forget_val,opt,device)
    elif method == 'neggrad':
        model_dic = basic_model_loader(opt, device)
        # ------------------------------dataloader--------------------------------------------------
        unlearn_dl = set_basic_loader(forget_train, opt)

        # ----------------------------Training Process--------------------------------
        Neggrad(model_dic=model_dic, unlearing_loader=unlearn_dl, device=device, opt=opt)

        # print(forget_train==forget_set['train'],len(forget_train))
        # ----------------------------Eva--------------------------------
        Evaluation(model_dic, retain_train, retain_val, forget_set['train'], forget_val, opt, device)
    elif method == "retrain":
        assert opt.saved_data_path != '', "Must retrain from saved data!!"
        retrain = False
        if retrain:
            param_path = 'SimCLR/runs/params.json'
            params = get_retrain_para(param_path)
            retain_indexes = list(range(0, 50000))
            train_loader = set_retrain_loader(opt.data_root, retain_indexes, params, mode="train")
            val_loader = set_retrain_loader(opt.data_root, retain_indexes, params, mode="val")
            retrain_model = retrain(params, train_loader, val_loader)
            model_dic = {'raw_model': retrain_model}
        else:
            retrain_model_path = 'SimCLR/runs/retrain_model/checkpoint_0300.pth.tar'
            raw_model_path = 'SimCLR/runs/original_model/checkpoint_0300.pth.tar'
            model_dic = model_loader(opt, device, raw_model_path, retrain_model_path)
        Evaluation(model_dic, retain_train, retain_val, forget_set['train'], forget_val, opt, device)
    # elif method == 'retrain':
    #     model_dic = basic_model_loader(opt, device)
    #     # ------------------------------dataloader--------------------------------------------------
    #     unlearn_dl = set_basic_loader(retain_train, opt)
    #     train_loader = torch.utils.data.DataLoader(retain_train, batch_size=opt.batch_size, shuffle=True,
    #                                                num_workers=opt.num_worker, pin_memory=True, sampler=None)
    #     val_loader = torch.utils.data.DataLoader(retain_val, batch_size=100, shuffle=False, num_workers=2,
    #                                              pin_memory=True)

    #     # ----------------------------Training Process--------------------------------
    #     Retrain(model_dic=model_dic, train_loader=train_loader,val_loader = val_loader, device=device, opt=opt)

    #     # ----------------------------Eva--------------------------------
    #     Evaluation(model_dic, retain_train, retain_val, forget_set['train'], forget_val, opt, device)

    elif method == 'scrub':
        model_dic = scrub_model_loader(opt, device)
        unlearn_dl = set_loader(retain_train, forget_train, opt)
        for i in range(epoches):
            epoch = i+1
            scrub(model_dic=model_dic, unlearing_loader=unlearn_dl, epoch=epoch, device=device, opt=opt)
        Evaluation(model_dic,retain_train, retain_val,forget_set['train'], forget_val,opt,device)

    time2 = time.time()
    print('Total time:',time2-time1)

if __name__ == '__main__':
    main()