import torch
from mu.bad_teaching import set_dataset, set_loader, bad_teaching, bad_te_model_loader
from mu.mu_utils import Evaluation
import mu.arg_parser as parser
from mu.Scrub import scrub, scrub_model_loader



def main():
    opt = parser.parse_option()
    method = opt.method
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoches = opt.epoches
    forget_set, retain_set = set_dataset(opt.data_name, opt.data_root, mode=opt.mode,
                                         forget_classes=opt.forget_class, forget_num=opt.forget_num)
    forget_train = forget_set['train']
    forget_val = forget_set['val']
    retain_train = retain_set['train']
    retain_val = retain_set['val']
#---------------------------------你俩和一下-------------------------------------------
    if method == 'bad_teaching':
        model_dic = bad_te_model_loader(opt, device)
        student = model_dic['student']
        compete_teacher = model_dic['compete_teacher']

        # ------------------------------dataloader--------------------------------------------------
        unlearn_dl = set_loader(retain_train, forget_train, opt)

        # ----------------------------Training Process--------------------------------
        bad_teaching(model_dic=model_dic, unlearing_loader=unlearn_dl, device=device, opt=opt)

        # ----------------------------Eva--------------------------------
        Evaluation(student,retain_train, retain_val,forget_train, forget_val,opt,device,competemodel=compete_teacher)
    elif method == 'scrub':
        model_dic = scrub_model_loader(opt, device)
        student = model_dic['student']
        compete_teacher = model_dic['compete_teacher']
        forget_set, retain_set = set_dataset(opt.data_name, opt.data_root, mode=opt.mode,
                                             forget_classes=opt.forget_class, forget_num=opt.forget_num)
        unlearn_dl = set_loader(retain_train, forget_train, opt)
        for i in range(epoches):
            epoch = i+1
            scrub(model_dic=model_dic, unlearing_loader=unlearn_dl, epoch=epoch, device=device, opt=opt)
        Evaluation(student,retain_train, retain_val,forget_train, forget_val,opt,device,competemodel=compete_teacher)

if __name__ == '__main__':
    main()