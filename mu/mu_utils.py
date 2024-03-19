import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .mu_metrics import MIA
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




