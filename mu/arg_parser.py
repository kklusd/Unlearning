import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for unlearning')
    parser.add_argument('--base_model', type=str, default='resnet18', help='basic model for classification')
    parser.add_argument('--teacher_path', type=str, default='./SimCLR/runs/original_model/checkpoint_0200.pth.tar',
                        help='teacher model path')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes in dataset')
    parser.add_argument('--sim_path', type=str, default='./SimCLR/runs/sim_model/checkpoint_0120.pth.tar',
                        help='simCLR model path')
    parser.add_argument('--out_dim', type=int, default=128, help='feature dim of simCLR')
    parser.add_argument('--lr', type=float, default=0.001, help='unlearning rate')
    parser.add_argument('--epoches', type=int, default=1, help='unlearning epoches')
    parser.add_argument('--method', type=str, default='bad_teaching', help='unlearning method')
    parser.add_argument('--mode', type=str, default='random', help='forget mode: classwise or random')
    parser.add_argument('--forget_num', type=int, default=100, help='size of forget set')
    parser.add_argument('--forget_class', type=int, default=0, help='forget class of classwise unlearning')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for unlearning')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for unlearing')
    parser.add_argument('--num_worker', type=int, default=6, help='number of workers for dataloader')
    parser.add_argument('--data_name', type=str, default='cifar10', help='dataset name')
    parser.add_argument('-data_root', type=str, default='./SimCLR/datasets', help='root of dataset')
    parser.add_argument('--loss_weight', type=float, default = 0.5, help='control the clr loss weight')

    opt = parser.parse_args()
    return opt
