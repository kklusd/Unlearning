import torch
from torchvision import transforms, datasets
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from .dataset import UnlearningData, ContrastiveUnlearningData
from tqdm import tqdm
from SimCLR.models.resnet_classifier import ResNetClassifier
from SimCLR.models.resnet_simclr import ResNetSimCLR
from .mu_models import Student
import torchvision.models as models
import copy
from .mu_utils import simple_contrast_loss, contrast_loss

np.random.seed(123)

def set_loader(retain_data, forget_data, opt):
    if opt.supervised_mode == "simple":
        unlearning_data = UnlearningData(retain_data=retain_data, forget_data=forget_data)
    elif opt.supervised_mode == "original":
        unlearning_data = ContrastiveUnlearningData(forget_data=forget_data, retain_data=retain_data, data_name=opt.data_name, n_views=2)
    else:
        raise ValueError(opt.supervised_mode)
    unlearning_loader = DataLoader(unlearning_data, batch_size=opt.batch_size, shuffle=True, 
                                   num_workers=opt.num_worker, pin_memory=True)
    return unlearning_loader

def set_dataset(data_name, root, mode='classwise', forget_classes=0, forget_num=0,require_index = False,augment = False):
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
        all_indexes = np.arange(0, len(train_ds),1,dtype=np.int16)
        np.random.shuffle(all_indexes)
        forget_indexes = all_indexes[:forget_num]
        retain_indexes = all_indexes[forget_num:]
        random_forget = {'train':[], 'val': []}
        random_retain = {'train': [], 'val': []}
        if require_index:
            return forget_indexes
        for index in forget_indexes:
            random_forget['train'].append(train_ds[index])

        for index in retain_indexes:
            random_retain['train'].append(train_ds[index])

        for index in forget_indexes:
            random_forget['val'].append(train_ds[index])

        #for index in retain_indexes:
            #random_retain['val'].append(train_ds[index])
        for img, label in val_ds:
            random_retain['val'].append((img, label))

        return random_forget, random_retain
    else:
        raise ValueError(mode)

def UnlearnLoss(class_logits, loss_contrast, labels, compete_teacher_logits, unlearn_teacher_logits, KL_temperature,loss_weight):
    labels = torch.unsqueeze(labels, dim=1)
    f_teacher_out = F.softmax(compete_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)
    overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
    student_class = F.log_softmax(class_logits / KL_temperature, dim=1)
    kl_loss = F.kl_div(student_class, overall_teacher_out,reduction = 'batchmean')
    final_loss = loss_weight*loss_contrast + 1*kl_loss
    return final_loss

def bad_te_model_loader(opt, device):
    num_class = opt.num_class
    out_dim = opt.out_dim
    base_model = opt.base_model
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
    checkpoint_sim = torch.load(opt.sim_path, map_location=device)
    student = Student(base_model=base_model, pro_dim=out_dim, num_class=num_class)
    student_state = copy.deepcopy(student.state_dict())
    for i in range(len(compete_teacher.state_dict().keys())):
        key_1 = list(compete_teacher.state_dict().keys())[i]
        key_2 = list(student.state_dict().keys())[i]
        student_state[key_2] = copy.deepcopy(compete_teacher.state_dict()[key_1])
    for i in range(1, 5):
        key_1 = list(checkpoint_sim['state_dict'].keys())[-i]
        key_2 = list(student.state_dict().keys())[-i]
        student_state[key_2] = copy.deepcopy(checkpoint_sim['state_dict'][key_1])
    student.load_state_dict(student_state)
    student.to(device)

    for k, v in student.named_parameters():
        if 'projection_head' in k.split('.'):
            v.requires_grad_(False)
    model_dic = {'student': student, 
                 'unlearning_teacher': unlearn_teacher,
                 'compete_teacher': compete_teacher}
    if opt.supervised_mode == "simple":
        simCLR = ResNetSimCLR(base_model=base_model, out_dim=out_dim)
        simCLR.load_state_dict(checkpoint_sim['state_dict'])
        simCLR.to(device)
        simCLR.eval()
        model_dic['simclr'] = simCLR
    return model_dic

def unlearning_step(model, model_dic, data_loader, optimizer, device, KL_temperature, loss_weight, supervised_mode):
    losses = []
    for batch in tqdm(data_loader, desc='training',leave=False):
        x, y = batch
        if supervised_mode == "original":
            batch_size = int(x[0].shape[0])
            x = torch.cat(x, dim=0)
        else:
            batch_size = int(x.shape[0])
        x, y = x.to(device), y.to(device)
        class_logits, student_sim_feature = model(x)
        with torch.no_grad():
            compete_teacher_logits = model_dic['compete_teacher'](x)
            unlearn_teacher_logits = model_dic['unlearning_teacher'](x)
            if supervised_mode == "simple":
                sim_features = model_dic['simclr'](x)
                loss_contrast = simple_contrast_loss(student_sim_feature, sim_features, y)
            elif supervised_mode == "original":
                loss_contrast = contrast_loss(student_sim_feature, y, batch_size, device, n_views=2, temperature=1)
            else:
                raise ValueError(supervised_mode)
        optimizer.zero_grad()
        loss = UnlearnLoss(class_logits[0:batch_size, :], loss_contrast, labels=y,
                           compete_teacher_logits=compete_teacher_logits[0:batch_size, :], 
                           unlearn_teacher_logits=unlearn_teacher_logits[0:batch_size, :], 
                           KL_temperature=KL_temperature,loss_weight = loss_weight)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def bad_teaching(model_dic, unlearing_loader, epoch, device,  opt):
    student = model_dic['student']
    optimizer = opt.optimizer
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(student.parameters(), lr = opt.lr)
    else:
        optimizer = torch.optim.SGD(student.parameters(), lr = opt.lr, momentum = 0.9, weight_decay = 5e-4)

    loss = unlearning_step(model=student, model_dic=model_dic, data_loader=unlearing_loader,
                            optimizer=optimizer, device=device, KL_temperature=1,
                            loss_weight = opt.loss_weight, supervised_mode=opt.supervised_mode)
    print("Epoch {} Unlearning Loss {}".format(epoch, loss))
