import argparse
import torchvision.models as models
from SimCLR.models.resnet_classifier import ResNetClassifier
from SimCLR.models.resnet_simclr import ResNetSimCLR
from mu.mu_models import Student
import torch
import copy
from torch.utils.data import DataLoader
from mu.bad_teaching import *
from mu.mu_utils import evaluate

def parse_option():
    parser = argparse.ArgumentParser('argument for unlearning')
    parser.add_argument('--base_model', type=str, default='resnet18', help='basic model for classification')
    parser.add_argument('--teacher_path', type=str, default='./SimCLR/runs/original_model/checkpoint_0200.pth.tar',
                        help='teacher model path')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes in dataset')
    parser.add_argument('--sim_path', type=str, default='./SimCLR/runs/sim_model/checkpoint_0020.pth.tar',
                        help='simCLR model path')
    parser.add_argument('--out_dim', type=int, default=128, help='feature dim of simCLR')
    parser.add_argument('--lr', type=float, default=0.001, help='unlearning rate')
    parser.add_argument('--epoches', type=int, default=1, help='unlearning epoches')
    parser.add_argument('--method', type=str, default='bad_teaching', help='unlearning method')
    parser.add_argument('--mode', type=str, default='classwise', help='forget mode: classwise or random')
    parser.add_argument('--forget_num', type=int, default=100, help='size of forget set')
    parser.add_argument('--forget_class', type=int, default=0, help='forget class of classwise unlearning')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for unlearning')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for unlearing')
    parser.add_argument('--num_worker', type=int, default=6, help='number of workers for dataloader')
    parser.add_argument('--data_name', type=str, default='cifar10', help='dataset name')
    parser.add_argument('-data_root', type=str, default='./SimCLR/datasets', help='root of dataset')
    opt = parser.parse_args()
    return opt

def main():
    print(1)
    opt = parse_option()
    method = opt.method
    num_class = opt.num_class
    out_dim = opt.out_dim
    base_model = opt.base_model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoches = opt.epoches
    if method == 'bad_teaching':
        if base_model == 'resnet18':
            unlearn_teacher = models.resnet18(num_classes = num_class, pretrained=False)
            unlearn_teacher.to(device)
            unlearn_teacher.eval()
        elif base_model == 'resnet50':
            unlearn_teacher = models.resnet18(num_classes = num_class, pretrained=False)
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

        for k, v in student.named_parameters():
            if 'projection_head' in k.split('.'):
                v.requires_grad_(False)
        model_dic = {'student': student, 
                     'unlearning_teacher': unlearn_teacher,
                     'simclr': simCLR,
                     'compete_teacher': compete_teacher}
        for i in range(epoches):
            epoch = i + 1
        bad_teaching(model_dic=model_dic, unlearing_loader=unlearn_dl, epoch=epoch, device=device, opt=opt)
        print('After unlearning epoch {} student forget'.format(epoch))
        print(evaluate(student, forget_val_dl, device))
        print('After unlearning epoch {} student retain')
        print(evaluate(student, retain_val_dl, device))

if __name__ == '__main__':
    main()