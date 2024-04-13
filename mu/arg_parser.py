import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for unlearning')
    parser.add_argument('--base_model', type=str, default='resnet18', help='basic model for classification')
    parser.add_argument('--teacher_path', type=str, default='./SimCLR/runs/original_model/checkpoint_0300.pth.tar',
                        help='teacher model path')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes in dataset')
    parser.add_argument('--sim_path', type=str, default='./SimCLR/runs/sim_model/checkpoint_0200.pth.tar',
                        help='simCLR model path')
    parser.add_argument('--out_dim', type=int, default=128, help='feature dim of simCLR')
    parser.add_argument('--lr', type=float, default=0.001, help='unlearning rate')
    parser.add_argument('--epoches', type=int, default=1, help='unlearning epoches')
    parser.add_argument('--method', type=str, default='salUN', help='unlearning method:bad_teaching|neggrad|')
    parser.add_argument('--mode', type=str, default='random', help='forget mode: classwise or random')
    parser.add_argument('--forget_num', type=int, default=5000, help='size of forget set')
    parser.add_argument('--forget_class', type=int, default=0, help='forget class of classwise unlearning')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for unlearning')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for unlearing')
    parser.add_argument('--num_worker', type=int, default=6, help='number of workers for dataloader')
    parser.add_argument('--data_name', type=str, default='cifar10', help='dataset name')
    parser.add_argument('--data_root', type=str, default='./SimCLR/datasets', help='root of dataset')
    parser.add_argument('--loss_weight', type=float, default = 0.5, help='control the clr loss weight')
    parser.add_argument('--supervised_mode', type=str, default="simple", help='simple: direct feature compare; original:augment image and then contrast learning ')
    parser.add_argument('--saved_data_path', type=str, default='mu/saved_data', help='saved instance-wise unlearning data')
    parser.add_argument('--data_augment', type=str, default='None', help='method to augment forget dataset')
    parser.add_argument('--augment_num', type=int, default=500, help='augment data amount')







    #-----------------------------------salUN------------------------------------
    parser.add_argument(
        "--data", type=str, default="./SimCLR/datasets", help="location of the data corpus"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")


    parser.add_argument("--num_classes", type=int, default=10)

    ##################################### Architecture ############################################
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="model architecture"
    )



    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")

    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers in dataloader"
    )
    parser.add_argument(
        "--save_dir",
        help="The directory used to save the trained models",
        default='salUN/run/un_models',
        type=str,
    )
    parser.add_argument("--model_path", type=str, default='run/0model_SA_best.pth.tar',
                        help="the path of original model")

    ##################################### Training setting #################################################
    parser.add_argument(
        "--epochs", default=1, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")



    ##################################### Unlearn setting #################################################
    parser.add_argument(
        "--unlearn", type=str, default="RL", help="method to unlearn"
    )
    parser.add_argument(
        "--unlearn_lr", default=0.01, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--unlearn_epochs",
        default=1,
        type=int,
        help="number of total epochs for unlearn to run",
    )

    parser.add_argument(
        "--num_indexes_to_replace",
        type=int,
        default=4500,
        help="Number of data to forget",
    )
    parser.add_argument(
        "--class_to_replace", type=int, default=None, help="Specific class to forget"
    )

    parser.add_argument(
        "--indexes_to_replace",
        type=list,
        default=None,
        help="Specific index data to forget",
    )
    parser.add_argument("--alpha", default=0.2, type=float, help="unlearn noise")
    parser.add_argument("--mask_path", default=None, type=str,
                        help="the path of saliency map")

    opt = parser.parse_args()
    return opt
