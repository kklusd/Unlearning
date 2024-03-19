import sys
import os
PROJ_DIR = 'D:/Nipstone/gittt/Unlearning-1'
sys.path.append(os.path.join(PROJ_DIR, 'mu'))
sys.path.append(PROJ_DIR)
import torch
from torchvision import transforms, datasets
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from dataset import UnlearningData
from tqdm import tqdm
from SimCLR.models.resnet_classifier import ResNetClassifier

from mu_models import Student
import torchvision.models as models
import copy

np.random.seed(123)


def set_loader(retain_data, forget_data, opt):
    unlearning_data = UnlearningData(retain_data=retain_data, forget_data=forget_data)
    unlearning_loader = DataLoader(unlearning_data, batch_size=opt.batch_size, shuffle=True,
                                   num_workers=opt.num_worker, pin_memory=True)
    return unlearning_loader


def set_dataset(data_name, root, mode='classwise', forget_classes=0, forget_num=0):
    if data_name == 'cifar10':
        size = 32
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([transforms.Resize(size),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])
        train_ds = datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
        val_ds = datasets.CIFAR10(root=root, train=False, transform=transform)
    elif data_name == 'stl10':
        size = 96
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([transforms.Resize(size),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])
        train_ds = datasets.STL10(root=root, split='train', transform=transform, download=True)
        val_ds = datasets.STL10(root=root, split='test', transform=transform)
    else:
        raise ValueError(data_name)

    if mode == 'classwise':
        assert forget_classes >= 0 and forget_classes < 10, 'must select 0-9 class to forget'
        classwise_forget = {'train': [], 'val': []}
        classwise_retain = {'train': [], 'val': []}
        for img, label in train_ds:
            if label == forget_classes:
                classwise_forget['train'].append((img, label))
            else:
                classwise_retain['train'].append((img, label))
        for img, label in val_ds:
            if label == forget_classes:
                classwise_forget['val'].append((img, label))
            else:
                classwise_retain['val'].append((img, label))
        return classwise_forget, classwise_retain
    elif mode == 'random':
        assert forget_num > 0 and forget_num < len(train_ds), 'must ensure forget_num is larger than 0'
        all_indexes = np.arange(0, len(train_ds), 1, dtype=np.int16)
        np.random.shuffle(all_indexes)
        forget_indexes = all_indexes[:forget_num]
        retain_indexes = all_indexes[forget_num:]
        random_forget = {'train': [], 'val': []}
        random_retain = {'train': [], 'val': []}

        for index in forget_indexes:
            random_forget['train'].append(train_ds[index])

        for index in retain_indexes:
            random_retain['train'].append(train_ds[index])

        for index in forget_indexes:
            random_forget['val'].append(train_ds[index])

        for index in retain_indexes:
            random_retain['val'].append(train_ds[index])

        return random_forget, random_retain
    else:
        raise ValueError(mode)


def UnlearnLoss(class_logits, student_sim_features, sim_features, labels, compete_teacher_logits,
                unlearn_teacher_logits, KL_temperature, loss_weight):
    student_sim = F.normalize(student_sim_features, dim=1)
    sim = F.normalize(sim_features, dim=1)
    similarity = torch.sum(sim * student_sim, dim=-1)
    adj_weight = 1 / (1 + np.e ** (1 - 2 * torch.count_nonzero(labels).item() / labels.numel()))
    # print(adj_weight)
    sim_loss = (1 - adj_weight) * torch.mean(labels * similarity) + adj_weight * torch.mean((labels - 1) * similarity)
    labels = torch.unsqueeze(labels, dim=1)
    f_teacher_out = F.softmax(compete_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_class = F.log_softmax(class_logits / KL_temperature, dim=1)
    kl_loss = F.kl_div(student_class, overall_teacher_out, reduction='batchmean')
    final_loss = loss_weight * sim_loss + 1 * kl_loss
    return final_loss


def basic_model_loader(opt, device):
    num_class = opt.num_class
    out_dim = opt.out_dim
    base_model = opt.base_model
    if base_model == 'resnet18':
        unlearn_teacher = models.resnet18(num_classes=num_class, weights=None)
        unlearn_teacher.to(device)
        unlearn_teacher.eval()
    elif base_model == 'resnet50':
        unlearn_teacher = models.resnet18(num_classes=num_class, weights=None)
        unlearn_teacher.to(device)
        unlearn_teacher.eval()
    else:
        raise ValueError(base_model)
    compete_teacher = ResNetClassifier(num_class=num_class, base_model=base_model)
    checkpoint_te = torch.load(opt.teacher_path, map_location=device)
    compete_teacher.load_state_dict(checkpoint_te['state_dict'])
    compete_teacher.to(device)
    compete_teacher.eval()
    student = Student(base_model=base_model, pro_dim=out_dim, num_class=num_class)
    student_state = copy.deepcopy(student.state_dict())
    for i in range(len(compete_teacher.state_dict().keys())):
        key_1 = list(compete_teacher.state_dict().keys())[i]
        key_2 = list(student.state_dict().keys())[i]
        student_state[key_2] = copy.deepcopy(compete_teacher.state_dict()[key_1])
    student.load_state_dict(student_state)
    student.to(device)

    for k, v in student.named_parameters():
        if 'projection_head' in k.split('.'):
            v.requires_grad_(False)
    model_dic = {'student': student,
                 'unlearning_teacher': unlearn_teacher,
                 'compete_teacher': compete_teacher}
    return model_dic


def unlearning_step(model, data_loader, optimizer, device, KL_temperature,loss_weight):
    losses = []
    for batch in tqdm(data_loader, desc='test', leave=False):
        # for batch in data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        class_logits = model(x)
        optimizer.zero_grad()
        loss = -1*torch.nn.CrossEntropyLoss()(class_logits, y)
        #loss = UnlearnLoss(class_logits,  labels=y, KL_temperature=KL_temperature,loss_weight=loss_weight)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def Neggrad(model_dic, unlearing_loader, device, opt):
    epoch = 0
    for i in range(opt.epoches):
        epoch = i + 1
        model = model_dic['unlearning_teacher']
        optimizer = opt.optimizer
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

        loss = unlearning_step(model=model, data_loader=unlearing_loader,
                               optimizer=optimizer, device=device, KL_temperature=1, loss_weight=opt.loss_weight)
        print("Epoch {} Unlearning Loss {}".format(epoch, loss))