import argparse
import torchvision.models as models
from SimCLR.models.resnet_classifier import ResNetClassifier
from SimCLR.models.resnet_simclr import ResNetSimCLR
from mu.mu_models import Student
import torch
import copy
from torch.utils.data import DataLoader
from mu.bad_teaching import *
from mu.bad_teaching import set_dataset
from mu.mu_utils import Evaluation
import time
from tqdm import trange,tqdm
import mu.arg_parser as parser


def main():
    opt = parser.parse_option()
    method = opt.method
    num_class = opt.num_class
    out_dim = opt.out_dim
    base_model = opt.base_model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoches = opt.epoches


    if method == 'bad_teaching':
        if base_model == 'resnet18':
            unlearn_teacher = models.resnet18(num_classes = num_class,  weights = None)
            unlearn_teacher.to(device)
            unlearn_teacher.eval()
        elif base_model == 'resnet50':
            unlearn_teacher = models.resnet18(num_classes = num_class, weights = None)
            unlearn_teacher.to(device)
            unlearn_teacher.eval()
        else:
            raise ValueError(base_model)
        compete_teacher = ResNetClassifier(num_class=num_class, base_model=base_model)
        checkpoint_te = torch.load(opt.teacher_path, map_location=device)
        compete_teacher.load_state_dict(checkpoint_te['state_dict'])
        compete_teacher.to(device)
        compete_teacher.eval()
        simCLR = ResNetSimCLR(base_model=base_model, out_dim=out_dim)
        checkpoint_sim = torch.load(opt.sim_path, map_location=device)
        simCLR.load_state_dict(checkpoint_sim['state_dict'])
        simCLR.to(device)
        simCLR.eval()
        student = Student(base_model=base_model, pro_dim=out_dim, num_class=num_class)
        student_state = copy.deepcopy(student.state_dict())
        for i in range(len(compete_teacher.state_dict().keys())):
            key_1 = list(compete_teacher.state_dict().keys())[i]
            key_2 = list(student.state_dict().keys())[i]
            student_state[key_2] = copy.deepcopy(compete_teacher.state_dict()[key_1])
        for i in range(1, 5):
            key_1 = list(simCLR.state_dict().keys())[-i]
            key_2 = list(student.state_dict().keys())[-i]
            student_state[key_2] = copy.deepcopy(simCLR.state_dict()[key_1])
        student.load_state_dict(student_state)
        student.to(device)

        for k, v in student.named_parameters():
            if 'projection_head' in k.split('.'):
                v.requires_grad_(False)
        model_dic = {'student': student, 
                     'unlearning_teacher': unlearn_teacher,
                     'simclr': simCLR,
                     'compete_teacher': compete_teacher}


        # ------------------------------dataloader--------------------------------------------------
        forget_set, retain_set = set_dataset(opt.data_name, opt.data_root, mode=opt.mode,
                                             forget_classes=opt.forget_class, forget_num=opt.forget_num)
        forget_train = forget_set['train']
        forget_val = forget_set['val']
        retain_train = retain_set['train']
        retain_val = retain_set['val']
        unlearn_dl = set_loader(retain_train, forget_train, opt)

        # ----------------------------Training Process--------------------------------
        for i in range(epoches):
            epoch = i + 1
            bad_teaching(model_dic=model_dic, unlearing_loader=unlearn_dl, epoch=epoch, device=device, opt=opt)

        # ----------------------------Eva--------------------------------
        Evaluation(student,retain_train, retain_val,forget_train, forget_val,opt,device,competemodel=compete_teacher)


if __name__ == '__main__':
    main()