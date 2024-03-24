import sys
import os
PROJ_DIR = 'D:/Nipstone/gittt/Unlearning-1'
sys.path.append(os.path.join(PROJ_DIR, 'mu'))
sys.path.append(PROJ_DIR)
import torch
import numpy as np
from tqdm import tqdm
from SimCLR.models.resnet_classifier import ResNetClassifier

np.random.seed(123)


def basic_model_loader(opt, device):
    num_class = opt.num_class
    out_dim = opt.out_dim
    base_model = opt.base_model
    raw_model = ResNetClassifier(num_class=num_class, base_model=base_model)
    checkpoint_te = torch.load(opt.teacher_path, map_location=device)
    raw_model.load_state_dict(checkpoint_te['state_dict'])
    raw_model.to(device)
    raw_model.eval()

    model_dic = {'raw_model': raw_model}
    return model_dic


def unlearning_step(model, data_loader, optimizer, device):
    losses = []
    for batch in tqdm(data_loader, desc='test', leave=False):
        # for batch in data_loader:
        x, y = batch
        x = torch.cat(x, dim=0)
        x, y = x.to(device), y.to(device)
        class_logits = model(x)
        optimizer.zero_grad()
        loss = -1*torch.nn.CrossEntropyLoss()(class_logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def Neggrad(model_dic, unlearing_loader, device, opt):
    epoch = 0
    for i in range(opt.epoches):
        epoch = i + 1
        model = model_dic['raw_model']
        optimizer = opt.optimizer
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

        loss = unlearning_step(model=model, data_loader=unlearing_loader,
                               optimizer=optimizer, device=device)
        print("Epoch {} Unlearning Loss {}".format(epoch, loss))