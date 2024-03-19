import torch
from torch.nn import functional as F
import numpy as np
import sys
import os
PROJ_DIR = 'D:/Nipstone/gittt/Unlearning-1'
sys.path.append(os.path.join(PROJ_DIR, 'mu'))
sys.path.append(PROJ_DIR)
from SimCLR.models.resnet_classifier import ResNetClassifier
from SimCLR.models.resnet_simclr import ResNetSimCLR
from mu_models import Student
import copy
    
# forget set 最大化老师和学生差距，retain set最小化 loss写在一起。 Cross entropy term = 0

def scrub_model_loader(opt, device):
    num_class = opt.num_class
    out_dim = opt.out_dim
    base_model = opt.base_model
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
                    'simclr': simCLR,
                    'compete_teacher': compete_teacher}
    return model_dic



def UnlearnLoss_scrub(class_logits, student_sim_features, sim_features, labels,
                        compete_teacher_logits, KL_temperature, loss_weight):
    student_sim = F.normalize(student_sim_features, dim=1)
    sim = F.normalize(sim_features, dim=1)
    similarity = torch.sum(sim*student_sim, dim=-1)
    adj_weight = 1 / (1 + np.e ** (1 - 2 * torch.count_nonzero(labels).item() / labels.numel()))
    sim_loss = (1 - adj_weight) * torch.mean(labels * similarity) + adj_weight * torch.mean((labels - 1) * similarity)
    labels = torch.unsqueeze(labels, dim=1)
    teacher_out = F.softmax(compete_teacher_logits / KL_temperature, dim=1)
    student_class = F.log_softmax(class_logits / KL_temperature, dim=1)
    
    kl_loss_forget = labels * F.kl_div(student_class, teacher_out,reduction = 'batchmean') 
    kl_loss_retain = (1-labels) * F.kl_div(student_class, teacher_out,reduction = 'batchmean')

    total_loss = kl_loss_retain - kl_loss_forget

    return total_loss + loss_weight*sim_loss


def unlearning_step_scrub(model, compete_teacher, simclr, data_loader, optimizer, device, KL_temperature, loss_weight):
    losses = []
    for batch in data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            compete_teacher_logits = compete_teacher(x)
            sim_features = simclr(x)

        class_logits, student_sim_feature = model(x)
        optimizer.zero_grad()
        loss = UnlearnLoss_scrub(class_logits, student_sim_feature, sim_features, labels=y, 
                                compete_teacher_logits=compete_teacher_logits,
                                KL_temperature=KL_temperature,loss_weight = loss_weight)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def scrub(model_dic, unlearing_loader, epoch, device,  opt):
    simclr = model_dic['simclr']
    compete_teacher = model_dic['compete_teacher']
    student = model_dic['student']
    simclr.eval()
    compete_teacher.eval()
    optimizer = opt.optimizer

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(student.parameters(), lr = opt.lr)
    else:
        optimizer = torch.optim.SGD(student.parameters(), lr = opt.lr, momentum = 0.9, weight_decay = 5e-4)

    loss = unlearning_step_scrub(model=student, compete_teacher=compete_teacher, 
                                simclr=simclr, data_loader=unlearing_loader,
                                optimizer=optimizer, device=device, KL_temperature=1)
    
    print("Epoch {} Unlearning Loss {}".format(epoch, loss))