import torch
from mu.bad_teaching import set_dataset, set_loader, bad_teaching, bad_te_model_loader
from mu.mu_utils import Evaluation
import mu.arg_parser as parser
from mu.Scrub import scrub, scrub_model_loader
from mu.mu_basic import Neggrad,basic_model_loader,set_basic_loader
import os
import pickle
from mu.mu_data import aug
def main():
    opt = parser.parse_option()
    method = opt.method
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoches = opt.epoches
    if opt.mode == 'random' and opt.saved_data_path != '':
        forget_data_file = os.path.join(opt.saved_data_path, 'forget_data.pt')
        retain_data_file = os.path.join(opt.saved_data_path, 'retain_data.pt')
        with open(forget_data_file, 'rb') as f:
                forget_set = pickle.load(f)
                f.close()
        with open(retain_data_file, 'rb') as f:
            retain_set = pickle.load(f)
            f.close()
    else:
        forget_set, retain_set = set_dataset(opt.data_name, opt.data_root, mode=opt.mode,
                                            forget_classes=opt.forget_class, forget_num=opt.forget_num)
    forget_train = forget_set['train']
    forget_val = forget_set['val']
    retain_train = retain_set['train']
    retain_val = retain_set['val']

#---------------------------------method-------------------------------------------
    if method == 'bad_teaching':
        model_dic = bad_te_model_loader(opt, device)
        student = model_dic['student']

        # ------------------------------dataloader--------------------------------------------------
        unlearn_dl = set_loader(retain_train, forget_train, opt)

        # ----------------------------Training Process--------------------------------
        for i in range(epoches):
            epoch = i + 1
            bad_teaching(model_dic=model_dic, unlearing_loader=unlearn_dl, epoch=epoch, device=device, opt=opt)

        # ----------------------------Eva--------------------------------
        Evaluation(model_dic,retain_train, retain_val,forget_train, forget_val,opt,device)
    elif method == 'neggrad':
        if True:
            forget_train2 = aug(forget_train)
        model_dic = basic_model_loader(opt, device)
        # ------------------------------dataloader--------------------------------------------------
        unlearn_dl = set_basic_loader(forget_train2, opt)

        # ----------------------------Training Process--------------------------------
        Neggrad(model_dic=model_dic, unlearing_loader=unlearn_dl, device=device, opt=opt)

        # ----------------------------Eva--------------------------------
        Evaluation(model_dic, retain_train, retain_val, forget_train, forget_val, opt, device)
    elif method == 'scrub':
        model_dic = scrub_model_loader(opt, device)
        student = model_dic['student']
        forget_set, retain_set = set_dataset(opt.data_name, opt.data_root, mode=opt.mode,
                                             forget_classes=opt.forget_class, forget_num=opt.forget_num)
        unlearn_dl = set_loader(retain_train, forget_train, opt)
        for i in range(epoches):
            epoch = i+1
            scrub(model_dic=model_dic, unlearing_loader=unlearn_dl, epoch=epoch, device=device, opt=opt)
        Evaluation(model_dic,retain_train, retain_val,forget_train, forget_val,opt,device)

if __name__ == '__main__':
    main()