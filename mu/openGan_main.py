import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import json
import os
from OpenGAN.feature_generate import features_generator
from bad_teaching import set_dataset
from OpenGAN.openGan_utils import FeaturesSet, model_init
import tqdm
import copy


parser = argparse.ArgumentParser('argument for OpenGAN')
parser.add_argument('--noise_size', type=int, default=100, help='noise size for generator')
parser.add_argument('--ots_size', type=int, default=512, help='size of OTS feature')
parser.add_argument('--ngf', type=int, default=64, help='size of feature map for generator')
parser.add_argument('--ndf', type=int, default=64, help='size of features map for discriminator')
parser.add_argument('--data_name', type=str, default='cifar10', help='dataset name')
parser.add_argument('-data_root', type=str, default='./SimCLR/datasets', help='root of dataset')
parser.add_argument('--batch_size', type=int, default=256, help='batch size for OpenGan')
parser.add_argument('--num_class', type=int, default=10, help='number of classes in dataset')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epoches', type=int, default=1, help='unlearning epoches')
parser.add_argument('--features_path', type=str, default='', help='path of stored features')
parser.add_argument('--data_path', type=str, default='', 
                    help='saved dataset for instance_wise unlearning')
parser.add_argument('--forget_num', type=int, default=512, help='number of forget data')
parser.add_argument('--base_model', type=str, default='resnet_18', help='base model for feature generator')
parser.add_argument('--state_dict_path', type=str, default='./SimCLR/runs/original_model/checkpoint_0200.pth.tar',
                    help='feature generator model checkpoint path')
parser.add_argument('--save_dir', type=str, default='OpenGAN_runs', help='Dirictory of model save')



def get_features(args, device):
    if args.features_path != '':
        forget_features_file = os.path.join(args.features_path, 'forget_features.pt')
        retain_features_file = os.path.join(args.features_path, 'retain`_features.pt')
        with open(forget_features_file, 'rb') as f:
            forget_features = pickle.load(f)
            f.close()
        with open(retain_features_file, 'rb') as f:
            retain_features = pickle.load(f)
            f.close()
    else:
        forget_data_file = os.path.join(args.data_path, 'forget_data.json')
        retain_data_file = os.path.join(args.data_path, 'retain_data.json')
        if args.data_path != '':
            with open(forget_data_file, 'r') as f:
                forget_set = json.load(f)
                f.close()
            with open(retain_data_file, 'r') as f:
                retain_set = json.load(f)
                f.close()
            forget_data = forget_set['train']
            retain_data = retain_set['train'][0:len(forget_data)]
        else:
            forget_set, retain_set = set_dataset(args.data_name,args.data_root, mode='random',
                                         forget_classes=0, forget_num=args.forget_num)
            with open(forget_data_file, 'w') as f:
                json.dump(forget_set, f) 
                f.close()
            with open(retain_data_file, 'w') as f:
                json.dump(retain_set, f)
                f.close()
            forget_data = forget_set['train']
            retain_data = retain_set['train'][0:len(forget_data)]
        new_feat_generator = features_generator(base_model=args.base_model, out_dim=128, 
                                             state_dict_path=args.state_ditc_path, device=device)
        forget_features, retain_features = new_feat_generator.generate(forget_data, retain_data, 
                                                                       batch_size=128, out_path='OpenGAN/saved_features')
    return forget_features, retain_features

def main():
    args = parser.parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    criterion = nn.BCELoss()
    forget_features, retain_features = get_features(args, device)
    feature_set = FeaturesSet(forget_features=forget_features, retain_features=retain_features)
    feature_loader = DataLoader(feature_set, batch_size=args.batch_size, shuffle=True, 
                                   num_workers=6, pin_memory=True)
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr/1.5, betas=(0.5, 0.999))
    optimizer_G = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    netG, netD = model_init(args, device)
    for epoch in args.epoches:
        iter = 0
        for batch in tqdm(feature_loader, desc='openGan',leave=False):
            features, y = batch
            features = features.to(device)
            y = y.to(device)
            netD.zero_grad()
            batch_size =features.shape[0]
            out_put = netD(features).view(-1)
            D_x_close = (y * out_put).mean().item()
            D_x_open = ((1-y) * out_put).mean().item()
            err_D_real = criterion(out_put, y)
            err_D_real.backward()
            noise = torch.randn(batch_size, args.noise_size, 1, 1).to(device)
            fake = netG(noise)
            label = torch.full((batch_size, ), 0).to(device)
            out_put = netD(fake.detach()).view(-1)
            D_G_z1 = out_put.mean().item()
            err_D_fake = criterion(out_put, label)
            err_D_fake.backward()
            err_D = err_D_real + err_D_fake
            optimizer_D.step()
            netG.zero_grad()
            label = label.fill_(1)
            out_put = netD(fake).view(-1)
            D_G_z2 = out_put.mean().item()
            err_G = criterion(out_put, label)
            err_G.backward()
            optimizer_G.step()
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x_close): %.4f\tD(x_open): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, args.epoches, iter, len(feature_loader),
                     err_D.item(), err_G.item(), D_x_close, D_x_open, D_G_z1, D_G_z2))
    cur_model_wts = copy.deepcopy(netG.state_dict())
    path_to_save_paramOnly = os.path.join(args.save_dir, 'epoch-{}.GNet'.format(epoch+1))
    torch.save(cur_model_wts, path_to_save_paramOnly)
    
    cur_model_wts = copy.deepcopy(netD.state_dict())
    path_to_save_paramOnly = os.path.join(args.save_dir, 'epoch-{}.DNet'.format(epoch+1))
    torch.save(cur_model_wts, path_to_save_paramOnly)
