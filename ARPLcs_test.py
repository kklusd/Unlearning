import argparse
import torch
import pickle
import os
from OpenGAN.network import Generator
from tqdm import tqdm
import copy
from OpenGAN.openGan_utils import t_sne_visial
from ARPL.models import gan
from OpenGAN.feature_generate import FeaturesGenerator
from SimCLR.models.resnet_classifier import ResNetClassifier
from mu.mu_models import BasicResnet

parser = argparse.ArgumentParser('argument for OpenGAN')
parser.add_argument('--noise_size', type=int, default=100, help='noise size for generator')
parser.add_argument('--ots_size', type=int, default=512, help='size of OTS feature')
parser.add_argument('--ngf', type=int, default=64, help='size of feature map for generator')
parser.add_argument('--batch_size', type=int, default=200, help='batch size for OpenGan')
parser.add_argument('--features_path', type=str, default='OpenGAN/saved_features', help='path of stored features')
parser.add_argument('--forget_num', type=int, default=5000, help='number of forget data')
parser.add_argument('--epoches', type=int, default=100, help='unlearning epoches')
parser.add_argument('--save_dir', type=str, default='log/models/un/resnet_un_ARPLoss_0.1_True_G.pth', help='Dirictory of model save')
parser.add_argument('--base_model', type=str, default='resnet18', help='base model for feature generator')
parser.add_argument('--state_dict_path', type=str, default='./SimCLR/runs/original_model/checkpoint_0200.pth.tar',
                    help='feature generator model checkpoint path')

def test():
    args = parser.parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    epoches = args.epoches
    model_path = args.save_dir
    trained_generator = gan._netG32(1,nz=100, ngf=64, nc=3)
    check_point = torch.load(model_path, map_location=device)
    trained_generator.load_state_dict(check_point,False)
    trained_generator.to(device)
    trained_generator.eval()
    #-------------------------因为ARPL生成的是图(3*32*32),再读一次feature---------------------------------
    base_model = args.base_model
    new_resnet = BasicResnet(base_model=base_model, out_dim=128)
    check_point = torch.load(args.state_dict_path, map_location=device)
    check_point_state = check_point['state_dict']
    base_model_state = new_resnet.state_dict()
    for i in range(len(base_model_state.keys())):
        key_1 = list(check_point_state.keys())[i]
        key_2 = list(base_model_state.keys())[i]
        base_model_state[key_2] = copy.deepcopy(check_point_state[key_1])
    new_resnet.load_state_dict(base_model_state)
    new_resnet.to(device)
    noise = torch.randn(3000, args.noise_size, 1, 1).to(device)
    fakedata = trained_generator(noise)
    generate_features = []
    for i in tqdm(range(5), desc='forget features generate', leave=False):
        noise = torch.randn(args.batch_size, args.noise_size, 1, 1).to(device)
        x = trained_generator(noise)
        features = new_resnet(x).detach().unsqueeze_(-1).unsqueeze_(-1)
        generate_features.append(features)
    fake_features = torch.cat(generate_features, dim=0).cpu()

    path = os.path.join(args.features_path, 'retain_features.pt')
    with open(path, 'rb') as f:
        retain_features = torch.load(f)
        f.close()
    path2 = os.path.join(args.features_path, 'forget_features.pt')
    with open(path2, 'rb') as f:
        forget_features = torch.load(f)
        f.close()
    all_fakes =  fake_features.squeeze(-1).squeeze(-1).numpy()
    retains = retain_features.squeeze(-1).squeeze(-1).numpy()
    forgets = forget_features.squeeze(-1).squeeze(-1).numpy()
    t_sne_visial(retains,forgets, all_fakes)

if __name__ == '__main__':
    test()


