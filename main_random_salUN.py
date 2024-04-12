import copy
import os
import time
from collections import OrderedDict
import pickle
from mu.bad_teaching import set_dataset
import salUN.arg_parser_salun
import salUN.evaluation as evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import salUN.unlearn as unlearn
import salUN.utils as utils
from salUN.trainer import validate
import mu.arg_parser as parser
from torch.utils.data import DataLoader
from salUN.trainer import train, validate
from mu.mu_retrain import *
from salUN.utils import setup_seed
from salUN.generate_mask import save_gradient_ratio




def main():

    mine = True

    opt = parser.parse_option()
    args = opt
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device(f"cuda:{0}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        salUN.utils.setup_seed(args.seed)


#-------------------------------------------------------------------------
    if opt.mode == 'random' and opt.saved_data_path != '':
        forget_data_file = os.path.join(opt.saved_data_path, 'forget_data.pt')
        retain_data_file = os.path.join(opt.saved_data_path, 'retain_data.pt')
        retain_indexes_file = os.path.join(opt.saved_data_path, 'retain_indexes.pt')
        with open(forget_data_file, 'rb') as f:
            forget_set = pickle.load(f)
            f.close()
        with open(retain_data_file, 'rb') as f:
            retain_set = pickle.load(f)
            f.close()
        with open(retain_indexes_file, 'rb') as f:
            retain_indexes = pickle.load(f)
            f.close()
    else:
        forget_set, retain_set, retain_indexes = set_dataset(opt.data_name, opt.data_root, mode=opt.mode,
                                                             forget_classes=opt.forget_class, forget_num=opt.forget_num)
    forget_train = copy.deepcopy(forget_set['train'])
    forget_val = forget_set['val']
    retain_train = retain_set['train']
    retain_val = retain_set['val']
    ret_loader = DataLoader(
        retain_train,
        batch_size=256,
        shuffle=True)
    for_loader = DataLoader(
        forget_train,
        batch_size=256,
        shuffle=True)
    vall_loader = DataLoader(
        retain_val,
        batch_size=256,
        shuffle=True)


    if mine:
        # model
        param_path = 'SimCLR/runs/params.json'
        params = get_retrain_para(param_path)
        raw_check_point = torch.load('./SimCLR/runs/original_model/checkpoint_0300.pth.tar', map_location='cuda:0')
        model = ResNetClassifier(params["arch"], num_class=opt.num_class, weights='IMAGENET1K_V1')
        model.load_state_dict(raw_check_point['state_dict'])
        model.cuda()

    #------------------Dataset----------------------------------


        unlearn_data_loaders = OrderedDict(
            retain=ret_loader, forget=for_loader, test=vall_loader
        )

        criterion = nn.CrossEntropyLoss()

        evaluation_result = None

        if opt.method == 'mask':

            save_gradient_ratio(unlearn_data_loaders, model, criterion, args)
    #-------------------------mask---------------------------------------

        if opt.method ==  'retrain':

            all_result = {}
            all_result["train_ta"] = []
            all_result["test_ta"] = []
            all_result["val_ta"] = []

            start_epoch = 0
            state = 0
            criterion = nn.CrossEntropyLoss()
            decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

            optimizer = torch.optim.SGD(
                model.parameters(),
                args.lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )
            for epoch in range(start_epoch, args.epochs):
                start_time = time.time()
                print(
                    "Epoch #{}, Learning rate: {}".format(
                        epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                    )
                )
                acc = train(ret_loader, model, criterion, optimizer, epoch, args)

                # evaluate on validation set
                tacc = validate(vall_loader, model, criterion, args)
                scheduler.step()

                all_result["train_ta"].append(acc)
                all_result["val_ta"].append(tacc)
                print(all_result)


        if opt.method == 'salUN':
            mask = torch.load(args.mask_path)
            unlearn_method = salUN.unlearn.get_unlearn_method(args.unlearn)
            unlearn_method(unlearn_data_loaders, model, criterion, args, mask)
            salUN.unlearn.save_unlearn_checkpoint(model, None, args)



        #-------------------------------------------------------------------------
        if evaluation_result is None:
            evaluation_result = {}

        if "new_accuracy" not in evaluation_result:
            accuracy = {}
            for name, loader in unlearn_data_loaders.items():
                #salUN.utils.dataset_convert_to_test(loader.dataset, args)
                val_acc = validate(loader, model, criterion, args)
                accuracy[name] = val_acc
                print(f"{name} acc: {val_acc}")

            evaluation_result["accuracy"] = accuracy

        for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
            if deprecated in evaluation_result:
                evaluation_result.pop(deprecated)

        """forget efficacy MIA:
            in distribution: retain
            out of distribution: test
            target: (, forget)"""
        if "SVC_MIA_forget_efficacy" not in evaluation_result:
            test_len = len(vall_loader)
            forget_len = len(forget_train)
            retain_len = len(retain_train)

            shadow_train = torch.utils.data.Subset(retain_train, list(range(test_len)))
            shadow_train_loader = torch.utils.data.DataLoader(
                shadow_train, batch_size=args.batch_size, shuffle=False
            )


            evaluation_result["SVC_MIA_forget_efficacy"] = salUN.evaluation.SVC_MIA(
                shadow_train=ret_loader,
                shadow_test=vall_loader,
                target_train=None,
                target_test=for_loader,
                model=model,
            )
            print(evaluation_result)
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(0))
            device = torch.device(f"cuda:{int(0)}")
        else:
            device = torch.device("cpu")

        os.makedirs(args.save_dir, exist_ok=True)
        if args.seed:
            utils.setup_seed(args.seed)
        seed = args.seed
        # prepare dataset
        (
            model,
            train_loader_full,
            val_loader,
            test_loader,
            marked_loader,
        ) = utils.setup_model_dataset(args)
        model.cuda()

        def replace_loader_dataset(
                dataset, batch_size=args.batch_size, seed=1, shuffle=True
        ):
            utils.setup_seed(seed)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=shuffle,
            )

        forget_dataset = copy.deepcopy(marked_loader.dataset)
        if args.dataset == "svhn":
            try:
                marked = forget_dataset.targets < 0
            except:
                marked = forget_dataset.labels < 0
            forget_dataset.data = forget_dataset.data[marked]
            try:
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
            except:
                forget_dataset.labels = -forget_dataset.labels[marked] - 1
            forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            try:
                marked = retain_dataset.targets >= 0
            except:
                marked = retain_dataset.labels >= 0
            retain_dataset.data = retain_dataset.data[marked]
            try:
                retain_dataset.targets = retain_dataset.targets[marked]
            except:
                retain_dataset.labels = retain_dataset.labels[marked]
            retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        else:
            try:
                marked = forget_dataset.targets < 0  # ---------------?-
                forget_dataset.data = forget_dataset.data[marked]
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
                forget_loader = replace_loader_dataset(
                    forget_dataset, seed=seed, shuffle=True
                )
                print(len(forget_dataset))
                retain_dataset = copy.deepcopy(marked_loader.dataset)
                marked = retain_dataset.targets >= 0
                retain_dataset.data = retain_dataset.data[marked]
                retain_dataset.targets = retain_dataset.targets[marked]
                retain_loader = replace_loader_dataset(
                    retain_dataset, seed=seed, shuffle=True
                )
                print(len(retain_dataset))
                assert len(forget_dataset) + len(retain_dataset) == len(
                    train_loader_full.dataset
                )
            except:
                marked = forget_dataset.targets < 0
                forget_dataset.imgs = forget_dataset.imgs[marked]
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
                forget_loader = replace_loader_dataset(
                    forget_dataset, seed=seed, shuffle=True
                )
                print(len(forget_dataset))
                retain_dataset = copy.deepcopy(marked_loader.dataset)
                marked = retain_dataset.targets >= 0
                retain_dataset.imgs = retain_dataset.imgs[marked]
                retain_dataset.targets = retain_dataset.targets[marked]
                retain_loader = replace_loader_dataset(
                    retain_dataset, seed=seed, shuffle=True
                )
                print(len(retain_dataset))
                assert len(forget_dataset) + len(retain_dataset) == len(
                    train_loader_full.dataset
                )

        print(f"number of retain dataset {len(retain_dataset)}")
        print(f"number of forget dataset {len(forget_dataset)}")
        unlearn_data_loaders = OrderedDict(
            retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
        )
        unlearn_data_loaders = OrderedDict(
            retain=ret_loader, forget=for_loader, test=vall_loader
        )
        criterion = nn.CrossEntropyLoss()

        evaluation_result = None

        if mine==True:
            if args.resume:
                checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

            if args.resume and checkpoint is not None:
                model, evaluation_result = checkpoint
            else:

                checkpoint = torch.load(args.model_path, map_location=device)
                if "state_dict" in checkpoint.keys():
                    checkpoint = checkpoint["state_dict"]

                if args.unlearn != "retrain":
                    model.load_state_dict(checkpoint, strict=False)

        if args.mask_path:
            mask = torch.load(args.mask_path)


        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_method(unlearn_data_loaders, model, criterion, args, mask)
        unlearn.save_unlearn_checkpoint(model, None, args)

        if evaluation_result is None:
            evaluation_result = {}

        if "new_accuracy" not in evaluation_result:
            accuracy = {}
            for name, loader in unlearn_data_loaders.items():
#                utils.dataset_convert_to_test(loader.dataset, args)
                val_acc = validate(loader, model, criterion, args)
                accuracy[name] = val_acc
                print(f"{name} acc: {val_acc}")

            evaluation_result["accuracy"] = accuracy
            unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

        for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
            if deprecated in evaluation_result:
                evaluation_result.pop(deprecated)

        """forget efficacy MIA:
            in distribution: retain
            out of distribution: test
            target: (, forget)"""
        if "SVC_MIA_forget_efficacy" not in evaluation_result:
            test_len = len(test_loader.dataset)
            forget_len = len(forget_dataset)
            retain_len = len(retain_dataset)

            utils.dataset_convert_to_test(retain_dataset, args)
            utils.dataset_convert_to_test(forget_loader, args)
            utils.dataset_convert_to_test(test_loader, args)

            shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
            shadow_train_loader = torch.utils.data.DataLoader(
                shadow_train, batch_size=args.batch_size, shuffle=False
            )

            evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
                shadow_train=shadow_train_loader,
                shadow_test=test_loader,
                target_train=None,
                target_test=forget_loader,
                model=model,
            )



if __name__ == "__main__":
    main()