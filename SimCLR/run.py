import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_utils.contrastive_learning_dataset import ContrastiveLearningDataset
from data_utils.supervised_dataset import SupervisedLearningDataset
from models.resnet_simclr import ResNetSimCLR
from models.resnet_classifier import ResNetClassifier
from simclr import SimCLR
from SupClassifier import SupClassifier
import json

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--framework', default='simclr', help='If training a supervised classifier, please type "supervised"')
parser.add_argument('--data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('--dataset_name', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--mo', type=float, default=0.9, help='momentum')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable_cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16_precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log_every_n_steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n_views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu_index', default=0, type=int, help='Gpu index.')


def main():
    args = parser.parse_args()
    with open("runs/params.json", mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
        f.close()
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    if args.framework == 'simclr':
        assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
        dataset = ContrastiveLearningDataset(args.data)

        train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)

        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                            last_epoch=-1)

        #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
        with torch.cuda.device(args.gpu_index):
            simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            simclr.train(train_loader)
    elif args.framework == 'supervised':
        dataset = SupervisedLearningDataset(args.data)
        train_dataset, val_dataset = dataset.get_dataset(args.dataset_name)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.workers, pin_memory=True, sampler=None)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)
        model = ResNetClassifier(base_model=args.arch, num_class=args.out_dim, weights='IMAGENET1K_V1')
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mo, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        with torch.cuda.device(args.gpu_index):
            classifier = SupClassifier(model=model, optimizer=optimizer, scheduler=scheduler, train_loader=train_loader, val_loader=val_loader, epochs=args.epochs)
            classifier.train()

if __name__ == "__main__":
    main()
