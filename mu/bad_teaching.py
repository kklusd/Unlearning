import torch
from torchvision import transforms, datasets
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from .dataset import UnlearningData
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
        val_ds = datasets.CIFAR10(root=root, split='test', transform=transform)
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
        all_indexes = np.arange(0, len(train_ds),1,dtype=np.int16)
        np.random.shuffle(all_indexes)
        forget_indexes = all_indexes[:forget_num]
        retain_indexes = all_indexes[forget_num:]
        random_forget = {'train':[], 'val': []}
        random_retain = {'train': [], 'val': []}
        
        for index in forget_indexes:
            random_forget['train'].append(train_ds[index])
        
        for index in retain_indexes:
            random_retain['train'].append(train_ds[index])

        for img, label in val_ds:
            random_retain['val'].append((img, label))
        return random_forget, random_retain
    else:
        raise ValueError(mode)

def UnlearnLoss(class_logits, student_sim_features, sim_features, labels, compete_teacher_logits, unlearn_teacher_logits, KL_temperature,loss_weight):
    student_sim = F.normalize(student_sim_features, dim=1)
    sim = F.normalize(sim_features, dim=1)
    sim_loss = torch.sum(-1 * labels * torch.sum(sim*student_sim, dim=-1))
    labels = torch.unsqueeze(labels, dim=1)
    f_teacher_out = F.softmax(compete_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)
    overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
    student_class = F.log_softmax(class_logits / KL_temperature, dim=1)
    kl_loss = F.kl_div(student_class, overall_teacher_out,reduction = 'batchmean')
    final_loss = loss_weight*sim_loss +1*kl_loss
    return final_loss


def unlearning_step(model, unlearning_teacher, compete_teacher, simclr, data_loader, optimizer, device, KL_temperature, loss_weight):
    losses = []
    for batch in data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            compete_teacher_logits = compete_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
            sim_features = simclr(x)
        class_logits, student_sim_feature = model(x)
        optimizer.zero_grad()
        loss = UnlearnLoss(class_logits, student_sim_feature, sim_features, labels=y, 
                           compete_teacher_logits=compete_teacher_logits, 
                           unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature,loss_weight = loss_weight)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def bad_teaching(model_dic, unlearing_loader, epoch, device,  opt):
    unlearning_teacher = model_dic['unlearning_teacher']
    simclr = model_dic['simclr']
    compete_teacher = model_dic['compete_teacher']
    student = model_dic['student']
    unlearning_teacher.eval()
    simclr.eval()
    compete_teacher.eval()
    optimizer = opt.optimizer
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(student.parameters(), lr = opt.lr)
    else:
        optimizer = torch.optim.SGD(student.parameters(), lr = opt.lr, momentum = 0.9, weight_decay = 5e-4)
    loss = unlearning_step(model=student, unlearning_teacher=unlearning_teacher, 
                            compete_teacher=compete_teacher, simclr=simclr, data_loader=unlearing_loader,
                            optimizer=optimizer, device=device, KL_temperature=1,loss_weight = opt.loss_weight)
    print("Epoch {} Unlearning Loss {}".format(epoch, loss))