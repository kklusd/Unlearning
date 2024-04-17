import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .mu_metrics import MIA
import numpy as np


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds==labels).item() / len(preds)) * 100  #100

def validation_step(model, batch, device):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    model = model.to(device)
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
    retain_train_dl = DataLoader(retain_train, opt.batch_size, opt.num_worker, pin_memory=True)
    forget_train_dl = DataLoader(forget_train, opt.batch_size, opt.num_worker, pin_memory=True)

    if opt.method == 'bad_teaching':
        model = model_dic['student']
        competemodel = model_dic['compete_teacher']

        Eva_Dr_before = evaluate(competemodel, retain_train_dl, device)
        Eva_Df_before = evaluate(competemodel, forget_train_dl, device)
        Eva_Dt_after = evaluate(model, retain_val_dl, device)
        Eva_Dr_after = evaluate(model, retain_train_dl, device)
        Eva_Df_after = evaluate(model, forget_train_dl, device)
        m1 = MIA(rt=retain_train, rv=retain_val, test=forget_train, model=model, method=opt.method)

        print('Before unlearning forget')
        print(Eva_Df_before)
        print('After unlearning epoch {} forget'.format(opt.epoches))
        print(Eva_Df_after)
        print('AccDr:{}'.format(Eva_Dr_after['Acc']))
        print('AccDf:{}'.format(Eva_Df_after['Acc']))
        print('AccDt:{}'.format(Eva_Dt_after['Acc']))
        print(Eva_Df_before['Acc'], Eva_Df_after['Acc'], Eva_Dr_before['Acc'], Eva_Dr_after['Acc'])
        print('Geo_metric:{}'.format(abs((Eva_Df_after['Acc'] - 95.77) * abs(Eva_Dt_after['Acc'] - 95.77))))
        print(m1)

    if opt.method == 'neggrad':
        model = model_dic['raw_model']
        competemodel = model_dic['compete_model']
        Eva_Dr_before = evaluate(competemodel, retain_train_dl,device)
        Eva_Df_before = evaluate(competemodel, forget_train_dl,device)
        Eva_Dt_after = evaluate(model, retain_val_dl,device)
        Eva_Dr_after = evaluate(model, retain_train_dl,device)
        Eva_Df_after = evaluate(model, forget_train_dl,device)
        m1 = MIA(rt=retain_train, rv=retain_val, test=forget_train, model=model,method = opt.method)

        print('Before unlearning forget')
        print(Eva_Df_before)
        print('After unlearning epoch {} forget'.format(opt.epoches))
        print(Eva_Dt_after)
        print('AccDr:{}'.format(Eva_Dr_after['Acc']))
        print('AccDf:{}'.format(Eva_Df_after['Acc']))
        print('AccDt:{}'.format(Eva_Dt_after['Acc']))
        print(Eva_Df_before['Acc'], Eva_Df_after['Acc'],Eva_Dr_before['Acc'],Eva_Dr_after['Acc'])
        print('Geo_metric:{}'.format((Eva_Df_after['Acc']-90)*(Eva_Dt_after['Acc']-80)))
        print(m1)

    if opt.method == 'retrain':
        raw_model = model_dic['raw_model']
        retrain_model = model_dic['retrain_model']
        print('Before unlearning forget')
        Eva_Dr_before = evaluate(raw_model, retain_train_dl,device)
        Eva_Df_before = evaluate(raw_model, forget_train_dl,device)
        print(Eva_Df_before)
        print('After unlearning epoch {} forget'.format(opt.epoches))
        Eva_Dt_after = evaluate(retrain_model, retain_val_dl,device)
        print(Eva_Dt_after)
        Eva_Dr_after = evaluate(retrain_model, retain_train_dl,device)
        Eva_Df_after = evaluate(retrain_model, forget_train_dl,device)

        print('AccDr:{}'.format(Eva_Dr_after['Acc']))
        print('AccDf:{}'.format(Eva_Df_after['Acc']))
        print('AccDt:{}'.format(Eva_Dt_after['Acc']))
        print(Eva_Df_before['Acc'],Eva_Df_after['Acc'] , Eva_Dr_before['Acc'], Eva_Dr_after['Acc'])
        print('Geo_metric:{}'.format((Eva_Df_before['Acc']-Eva_Df_after['Acc'])*(Eva_Dr_after['Acc']-Eva_Dr_before['Acc'])))
        m1 = MIA(rt=retain_train, rv=retain_val, test=forget_train, model=retrain_model,method = opt.method)
        print(m1)
        data = {'AccDr:{}':Eva_Dr_after['Acc'],'AccDf:{}':Eva_Df_after['Acc'],'AccDt:{}':Eva_Dt_after['Acc'],"MIA":m1}
        np.save('mu/saved_data/retrain_5000.npy', data)

    if opt.method == 'scrub':
        model = model_dic['student']
        competemodel = model_dic['compete_teacher']

        #--------------------------------------------------------------
        model = model_dic['student']
        competemodel = model_dic['compete_teacher']

        Eva_Dr_before = evaluate(competemodel, retain_train_dl, device)
        Eva_Df_before = evaluate(competemodel, forget_train_dl, device)
        Eva_Dt_after = evaluate(model, retain_val_dl, device)
        Eva_Dr_after = evaluate(model, retain_train_dl, device)
        Eva_Df_after = evaluate(model, forget_train_dl, device)
        m1 = MIA(rt=retain_train, rv=retain_val, test=forget_train, model=model, method=opt.method)

        print('Before unlearning forget')
        print(Eva_Df_before)
        print('After unlearning epoch {} forget'.format(opt.epoches))
        print(Eva_Dt_after)
        print('AccDr:{}'.format(Eva_Dr_after['Acc']))
        print('AccDf:{}'.format(Eva_Df_after['Acc']))
        print('AccDt:{}'.format(Eva_Dt_after['Acc']))
        print(Eva_Df_before['Acc'], Eva_Df_after['Acc'], Eva_Dr_before['Acc'], Eva_Dr_after['Acc'])
        print('Geo_metric:{}'.format(abs((Eva_Df_after['Acc'] - 95.77) * abs(Eva_Dt_after['Acc'] - 95.77))))
        print(m1)


    if opt.method == 'salUN':
            model = model_dic['raw_model']
            m1 = MIA(rt=retain_train, rv=retain_val, test=forget_train, model=model,method = opt.method)
            print(m1)

def contrast_loss(features, set_labels, batch_size, device, n_views, temperature):
    criterion = torch.nn.CrossEntropyLoss().to(device)
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
    # print(positives)
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    set_labels = set_labels.repeat(n_views)
    set_labels = set_labels.view(set_labels.shape[0], 1)
    # print("set_labels", set_labels.shape)
    positives = -set_labels * positives
    logits = torch.cat([positives, negatives], dim=1)
    # logits = -set_labels * logits
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return criterion(logits, labels)
    # return torch.sum(positives)

def simple_contrast_loss(student_sim_features, sim_features, set_labels):
    student_sim = F.normalize(student_sim_features, dim=1)
    sim = F.normalize(sim_features, dim=1)
    similarity = torch.sum(sim*student_sim, dim=-1)
    adj_weight = 1 / (1 + np.e ** (1 - 2 * torch.count_nonzero(set_labels).item() / set_labels.numel()))
    sim_loss = (1 - adj_weight) * torch.mean(set_labels * similarity) + adj_weight * torch.mean((set_labels - 1) * similarity)

    return sim_loss


def feature_visialization(model_dict, ul_loader, ul_method, device, visial_unlearn=True):
    def t_sne_visial(retain_features,forget_features, save_name):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        retain_fea_len = retain_features.shape[0]
        forget_fea_len = forget_features.shape[0]
        features = np.concatenate([retain_features,forget_features], axis=0)
        tsne_result = TSNE(n_components=2).fit_transform(features)
        def scale_to_01_range(x):
            value_range = (np.max(x) - np.min(x))
            starts_from_zero = x - np.min(x)
            return starts_from_zero / value_range
        tsne_x = scale_to_01_range(tsne_result[:,0])
        tsne_y = scale_to_01_range(tsne_result[:,1])
        colors = ['b', 'c']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(tsne_x[0:retain_fea_len], tsne_y[0:retain_fea_len], c=colors[0], label='retain',s=4)
        ax.scatter(tsne_x[retain_fea_len:retain_fea_len+forget_fea_len], tsne_y[retain_fea_len:retain_fea_len+forget_fea_len], c=colors[1], label='forget',s=2)
        ax.legend(loc='best')
        plt.savefig(save_name)
    
    if visial_unlearn:
        from .mu_models import BasicResnet
        import copy
        from tqdm import tqdm
        def get_feature_extractor(model):
            extractor = BasicResnet(base_model="resnet18", out_dim=10)
            model_state_dict = model.state_dict()
            extractor_state_dict = extractor.state_dict()
            for i in range(len(extractor_state_dict.keys())):
                key_1 = list(extractor_state_dict.keys())[i]
                key_2 = list(model_state_dict.keys())[i]
                extractor_state_dict[key_1] = copy.deepcopy(model_state_dict[key_2])
            extractor.load_state_dict(extractor_state_dict)
            extractor.to(device)
            return extractor
        if ul_method == "bad_teaching" or ul_method == "scrub":
            ul_model = model_dict["student"]
            ul_model = get_feature_extractor(ul_model)
        elif ul_method == "retrain":
            ul_model = model_dict["retrain_model"]
        else:
            ul_model = model_dict["raw_model"]
        with torch.no_grad():
            ul_model.eval()
            forget_features = []
            retain_features = []
            for batch in tqdm(ul_loader, desc='generating features',leave=False):
                x, y = batch
                x = x.to(device)
                output = ul_model(x)
                if len(output) == 2:
                    features = output[1].detach().cpu()
                else:
                    features = output.detach().cpu()
                for i in range(y.shape[0]):
                    if y[i] == 1:
                        forget_features.append(features[i])
                    else:
                        retain_features.append(features[i])
            forget_features = torch.stack(forget_features, dim=0).squeeze(-1).squeeze(-1).numpy()
            retain_features = torch.stack(retain_features, dim=0).squeeze(-1).squeeze(-1).numpy()
        t_sne_visial(retain_features, forget_features, save_name="unlearning_tsne.png")
                
    else:
        forget_feature_path = "OpenGAN/saved_features/forget_features.pt"
        retain_feature_path = "OpenGAN/saved_features/retain_features.pt"
        with open(forget_feature_path, 'rb') as f:
            forget_features = torch.load(f)
            f.close()
        with open(retain_feature_path, 'rb') as f:
            retain_features = torch.load(f)
            f.close()
        forget_features = forget_features.squeeze(-1).squeeze(-1).numpy()
        retain_features = retain_features.squeeze(-1).squeeze(-1).numpy()
        t_sne_visial(retain_features, forget_features, save_name="original_tsne.png")