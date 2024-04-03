import argparse
import torch
import pickle
import os
from OpenGAN.network import Generator
from tqdm import tqdm
import copy
from OpenGAN.openGan_utils import t_sne_visial

parser = argparse.ArgumentParser('argument for OpenGAN')
parser.add_argument('--noise_size', type=int, default=256, help='noise size for generator')
parser.add_argument('--ots_size', type=int, default=512, help='size of OTS feature')
parser.add_argument('--ngf', type=int, default=64, help='size of feature map for generator')
parser.add_argument('--batch_size', type=int, default=200, help='batch size for OpenGan')
parser.add_argument('--features_path', type=str, default='OpenGAN/saved_features', help='path of stored features')
parser.add_argument('--forget_num', type=int, default=5000, help='number of forget data')
parser.add_argument('--epoches', type=int, default=100, help='unlearning epoches')
parser.add_argument('--save_dir', type=str, default='OpenGAN/OpenGAN_runs', help='Dirictory of model save')

def test():
    args = parser.parse_args()
    device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')
    epoches = args.epoches
    model_path = os.path.join(args.save_dir, 'epoch-'+'{}'.format(epoches)+'.GNet')
    trained_generator = Generator(nz=args.noise_size, ngf=args.ngf, nc=args.ots_size)
    trained_generator.to(device)
    check_point = torch.load(model_path, map_location=device)
    trained_generator.load_state_dict(check_point)
    trained_generator.eval()
    forget_feature_path = os.path.join(args.features_path, 'forget_features.pt')
    with open(forget_feature_path, 'rb') as f:
        forget_features = torch.load(f)
        f.close()
    batch_size = args.batch_size
    all_fakes = []
    for i in tqdm(range(args.forget_num // batch_size)):
        noise = torch.randn(batch_size, args.noise_size, 1, 1).to(device)
        fake = trained_generator(noise)
        all_fakes.append(fake)
    all_fakes = torch.cat(all_fakes, dim=0).detach().squeeze(-1).squeeze(-1).cpu().numpy()
    reals = forget_features.squeeze(-1).squeeze(-1).numpy()
    t_sne_visial(reals, all_fakes)

if __name__ == '__main__':
    test()


