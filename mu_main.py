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
from mu.mu_utils import evaluate
from mu.mu_metrics import SVC_MIA
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
        #-------------------------------------
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

        forget_set, retain_set = set_dataset(opt.data_name, opt.data_root, mode=opt.mode, 
                                             forget_classes=opt.forget_class, forget_num=opt.forget_num)
        
        forget_train = forget_set['train']
        forget_val = forget_set['val']
        retain_train = retain_set['train']
        retain_val = retain_set['val']
        unlearn_dl = set_loader(retain_train, forget_train, opt)
        if opt.mode == 'classwise':
            forget_val_dl = DataLoader(forget_val, opt.batch_size, opt.num_worker, pin_memory=True)
        else:
            forget_val_dl = DataLoader(forget_train, opt.batch_size, opt.num_worker, pin_memory=True)
        retain_val_dl = DataLoader(retain_val, opt.batch_size, opt.num_worker, pin_memory=True)
        print('Before unlearning teacher forget')
        print(evaluate(compete_teacher, forget_val_dl, device))
        print('Before unlearning teacher retain')
        print(evaluate(compete_teacher, retain_val_dl, device))
        
        print('Before unlearning student forget')
        print(evaluate(student, forget_val_dl, device))
        print('Before unlearning student retain')
        print(evaluate(student, retain_val_dl, device))

#----------------------------Training Process--------------------------------
        for k, v in student.named_parameters():
            if 'projection_head' in k.split('.'):
                v.requires_grad_(False)
        model_dic = {'student': student, 
                     'unlearning_teacher': unlearn_teacher,
                     'simclr': simCLR,
                     'compete_teacher': compete_teacher}


        for i in trange(epoches):
            epoch = i + 1
            bad_teaching(model_dic=model_dic, unlearing_loader=unlearn_dl, epoch=epoch, device=device, opt=opt)
        print('After unlearning epoch {} student forget'.format(epoch))
        print(evaluate(student, forget_val_dl, device))
        print('After unlearning epoch {} student retain'.format(epoch))
        print(evaluate(student, retain_val_dl, device))


        #------------------other metrics----------------------
        """forget efficacy MIA:
            in distribution: retain
            out of distribution: test
            target: (, forget)
            train data:label1 ;val data:label0
            wish forget val label goes to 0 :即被认为没有参与训练
            MIA函数进行了1-mean，结果越靠近1越好"""
        test_len = 200

        shadow_train = torch.utils.data.Subset(retain_train, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=opt.batch_size, shuffle=False
        )
        shadow_test = torch.utils.data.Subset(retain_val, list(range(test_len)))
        shadow_test_loader = torch.utils.data.DataLoader(
            shadow_test, batch_size=opt.batch_size, shuffle=False
        )
        target_test = torch.utils.data.Subset(forget_val, list(range(test_len)))
        target_test_loader = torch.utils.data.DataLoader(
            target_test, batch_size=opt.batch_size, shuffle=False
        )

        SVC_MIA_forget_efficacy = SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=shadow_test_loader,
            target_train=None,
            target_test=target_test_loader,
            model=student,
        )
        print(SVC_MIA_forget_efficacy)

if __name__ == '__main__':
    main()