import torch
import numpy as np
from tqdm import tqdm
from SimCLR.models.resnet_classifier import ResNetClassifier
from torch.utils.data import DataLoader

from .mu_models import BasicClassifier
np.random.seed(123)
from .dataset import UnlearningData, BasicUnlearningData

def set_basic_loader(forget_data, opt):
    unlearning_data = BasicUnlearningData(forget_data=forget_data)
    unlearning_loader = DataLoader(unlearning_data, batch_size=opt.batch_size, shuffle=True,
                                   num_workers=opt.num_worker, pin_memory=True)
    return unlearning_loader
def basic_model_loader(opt, device):
    num_class = opt.num_class
    out_dim = opt.out_dim
    base_model = opt.base_model
    raw_model = ResNetClassifier(num_class=num_class, base_model=base_model)
    checkpoint_te = torch.load(opt.teacher_path, map_location=device)
    raw_model.load_state_dict(checkpoint_te['state_dict'])
    raw_model.to(device)
    competemodel = ResNetClassifier(num_class=num_class, base_model=base_model)
    competemodel.load_state_dict(checkpoint_te['state_dict'])
    competemodel.to(device)
    competemodel.eval()
    model_dic = {'raw_model': raw_model,'compete_model': competemodel}
    return model_dic


def unlearning_step(model, data_loader, optimizer, device):
    losses = []
    for batch in tqdm(data_loader, desc='test', leave=False):
        # for batch in data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        class_logits = model(x)
        optimizer.zero_grad()
        loss= -0.1*torch.nn.functional.cross_entropy(class_logits,y)
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
            optimizer = torch.optim.Adam(model.parameters(), lr=1)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.9, weight_decay=5e-4)

        loss = unlearning_step(model=model, data_loader=unlearing_loader,
                               optimizer=optimizer, device=device)
        print("Epoch {} Unlearning Loss {}".format(epoch, loss))