import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .mu_metrics import MIA
import numpy as np

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds==labels).item() / len(preds)) * 100

def validation_step(model, batch, device):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out = model(images)
    if type(out) == tuple:
        out = out[0]
    loss = F.cross_entropy(out, labels)
    acc = accuracy(out, labels)
    return {'Loss': loss.detach(), 'Acc': acc}


def validation_epoch_end(outputs):
    batch_losses = [x['Loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['Acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'Loss': epoch_loss.item(), 'Acc': epoch_acc.item()}

def evaluate(model, val_loader, device):
    model.eval()
    outputs = [validation_step(model, batch, device) for batch in val_loader]

    return validation_epoch_end(outputs)

def Evaluation(model_dic,retain_train,retain_val,forget_train,forget_val,opt,device):
    if opt.mode == 'classwise':
        forget_val_dl = DataLoader(forget_val, opt.batch_size, opt.num_worker, pin_memory=True)
    else:
        forget_val_dl = DataLoader(forget_train, opt.batch_size, opt.num_worker, pin_memory=True)
    retain_val_dl = DataLoader(retain_val, opt.batch_size, opt.num_worker, pin_memory=True)
    if opt.method == 'bad_teaching':
        model = model_dic['student']
        competemodel = model_dic['compete_teacher']

        #--------------------------------------------------------------
        print('Before unlearning teacher forget')
        print(evaluate(competemodel, forget_val_dl, device))
        print('Before unlearning teacher retain')
        print(evaluate(competemodel, retain_val_dl, device))

        print('Before unlearning student forget')
        print(evaluate(model, forget_val_dl, device))
        print('Before unlearning student retain')
        print(evaluate(model, retain_val_dl, device))

        print('After unlearning epoch {} student forget'.format(opt.epoches))
        print('After unlearning student forget')
        print(evaluate(model, forget_val_dl, device))
        print('After unlearning student retain')
        print(evaluate(model, retain_val_dl, device))
        # ------------------other metrics----------------------
        m1 = MIA(rt = retain_train, rv = retain_val, test = forget_val, model=model)
        m2 = MIA(rt = retain_train, rv = retain_val, test = forget_train, model=model)
        print(m1)
        print(m2)

    if opt.method == 'neggrad':
        model = model_dic['raw_model']
        print('Before unlearning teacher forget')
        print(evaluate(model, forget_val_dl, device))
        print('After unlearning epoch {} student forget'.format(opt.epoches))
        print('After unlearning student forget')
        print(evaluate(model, forget_val_dl, device))

        # ------------------other metrics----------------------
        m1 = MIA(rt=retain_train, rv=retain_val, test=forget_val, model=model)
        print(m1)




    print('Before unlearning teacher forget')
    print(evaluate(competemodel, forget_val_dl, device))
    print('Before unlearning teacher retain')
    print(evaluate(competemodel, retain_val_dl, device))

    print('Before unlearning student forget')
    print(evaluate(model, forget_val_dl, device))
    print('Before unlearning student retain')
    print(evaluate(model, retain_val_dl, device))

    print('After unlearning epoch {} student forget'.format(opt.epoches))
    print(evaluate(model, forget_val_dl, device))
    print(evaluate(model, retain_val_dl, device))
    # ------------------other metrics----------------------
    m1 = MIA(retain_train, retain_val, forget_val, model=model)
    m2 = MIA(retain_train, retain_val, forget_train, model=model)
    print(m1)
    print(m2)

def contrast_loss(features, set_labels, batch_size, device, n_views, temperature):
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits = -1 * set_labels * logits
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return torch.nn.CrossEntropyLoss(logits, labels)

def simple_contrast_loss(student_sim_features, sim_features, set_labels):
    student_sim = F.normalize(student_sim_features, dim=1)
    sim = F.normalize(sim_features, dim=1)
    similarity = torch.sum(sim*student_sim, dim=-1)
    adj_weight = 1 / (1 + np.e ** (1 - 2 * torch.count_nonzero(set_labels).item() / set_labels.numel()))
    sim_loss = (1 - adj_weight) * torch.mean(set_labels * similarity) + adj_weight * torch.mean((set_labels - 1) * similarity)
    return sim_loss
