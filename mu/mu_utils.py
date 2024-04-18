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
    positives = (1-set_labels) * positives - set_labels * positives
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


def feature_visialization(model_dict, retain_data, forget_data, ul_method, device, visial_unlearn=True):
    retain_train_dl = DataLoader(retain_data, 128, 4, pin_memory=True)
    forget_train_dl = DataLoader(forget_data, 128, 4, pin_memory=True)
    from tqdm import tqdm
    def t_sne_visial(retain_features, forget_features, save_name):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        all_feature_ls = []
        for label in retain_features:
            all_feature_ls.append(retain_features[label])
        for label in forget_features:
            all_feature_ls.append(forget_features[label])
        features = np.concatenate(all_feature_ls, axis=0)
        tsne_result = TSNE(n_components=2).fit_transform(features)
        def scale_to_01_range(x):
            value_range = (np.max(x) - np.min(x))
            starts_from_zero = x - np.min(x)
            return starts_from_zero / value_range
        tsne_x = scale_to_01_range(tsne_result[:,0])
        tsne_y = scale_to_01_range(tsne_result[:,1])
        del(features)
        colors = {"0":'#e50000', "1":'#653700', "2":'#7e1e9c', "3":'#15b01a', "4":'#00035b', "5":'#033500', "6":'#f97306', "7":'#13eac9', "8":'#029386', "9":'#06c2ac'}
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        for label in retain_features:
            feature_num = int(retain_features[label].shape[0])
            feature_indexes = np.arange(0, feature_num)
            current_x  = tsne_x[feature_indexes]
            tsne_x = np.delete(tsne_x, feature_indexes, axis=0)
            current_y = tsne_y[0:feature_num]
            tsne_y = np.delete(tsne_y, feature_indexes, axis=0)
            ax.scatter(current_x, current_y, c=colors[label], label=label, s=0.01)
        for label in forget_features:
            feature_num = int(forget_features[label].shape[0])
            feature_indexes = np.arange(0, feature_num)
            current_x  = tsne_x[feature_indexes]
            tsne_x = np.delete(tsne_x, feature_indexes, axis=0)
            current_y = tsne_y[0:feature_num]
            tsne_y = np.delete(tsne_y, feature_indexes, axis=0)
            ax.scatter(current_x, current_y, c=colors[label], label=label, marker="D", s=4)
        plt.axis('off')
        plt.savefig(save_name, dpi=100)
    
    if visial_unlearn:
        from .mu_models import BasicResnet
        import copy
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
            forget_features = {}
            retain_features = {}
            for batch in tqdm(retain_train_dl, desc='generating features',leave=False):
                x, y = batch
                x = x.to(device)
                output = ul_model(x)
                if len(output) == 2:
                    features = output[1].detach().cpu()
                else:
                    features = output.detach().cpu()
                for i in range(y.shape[0]):
                    label = str(y[i].item())
                    if label in retain_features:
                        retain_features[label].append(features[i])
                    else:
                        retain_features[label] = [features[i]]
            for batch in tqdm(forget_train_dl, desc='generating features',leave=False):
                x, y = batch
                x = x.to(device)
                output = ul_model(x)
                if len(output) == 2:
                    features = output[1].detach().cpu()
                else:
                    features = output.detach().cpu()
                for i in range(y.shape[0]):
                    label = str(y[i].item())
                    if label in forget_features:
                        forget_features[label].append(features[i])
                    else:
                        forget_features[label] = [features[i]]
            for label in forget_features:
                forget_features[label] = torch.stack(forget_features[label], dim=0).squeeze(-1).squeeze(-1).numpy()
            for label in retain_features:
                retain_features[label] = torch.stack(retain_features[label], dim=0).squeeze(-1).squeeze(-1).numpy()
        t_sne_visial(retain_features, forget_features, save_name="unlearning_tsne.pdf")
                
    else:
        from SimCLR.models.resnet_classifier import ResNetClassifier
        original_model = ResNetClassifier(base_model="resnet18", num_class=10, weights=None)
        check_point_path = "SimCLR/runs/original_model/checkpoint_0300.pth.tar"
        check_point = torch.load(check_point_path, map_location=device)
        original_model.load_state_dict(check_point['state_dict'])
        original_model.to(device)
        with torch.no_grad():
            original_model.eval()
            forget_features = {}
            retain_features = {}
            for batch in tqdm(retain_train_dl, desc='generating features',leave=False):
                x, y = batch
                x = x.to(device)
                output = original_model(x)
                if len(output) == 2:
                    features = output[1].detach().cpu()
                else:
                    features = output.detach().cpu()
                for i in range(y.shape[0]):
                    label = str(y[i].item())
                    if label in retain_features:
                        retain_features[label].append(features[i])
                    else:
                        retain_features[label] = [features[i]]
            for batch in tqdm(forget_train_dl, desc='generating features',leave=False):
                x, y = batch
                x = x.to(device)
                output = original_model(x)
                if len(output) == 2:
                    features = output[1].detach().cpu()
                else:
                    features = output.detach().cpu()
                for i in range(y.shape[0]):
                    label = str(y[i].item())
                    if label in forget_features:
                        forget_features[label].append(features[i])
                    else:
                        forget_features[label] = [features[i]]
            for label in forget_features:
                forget_features[label] = torch.stack(forget_features[label], dim=0).squeeze(-1).squeeze(-1).numpy()
            for label in retain_features:
                retain_features[label] = torch.stack(retain_features[label], dim=0).squeeze(-1).squeeze(-1).numpy()
        t_sne_visial(retain_features, forget_features, save_name="original_tsne.pdf")