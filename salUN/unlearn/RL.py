import time
from copy import deepcopy
import random
import numpy as np
import torch
import salUN.utils as utils

from .impl import iterative_unlearn


@iterative_unlearn
def RL(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    start = time.time()
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]

    if args.dataset == "cifar10" or args.dataset == "svhn":
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()

        # switch to train mode
        model.train()

        loader_len = len(forget_loader) + len(retain_loader)

        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer,
                            one_epoch_step=loader_len, args=args)
        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            target = torch.randint(0, args.num_classes, target.shape).cuda()

            # compute output
            output_clean, _ = model(image)
            loss = criterion(output_clean, target.to(torch.int64))

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

        for i, (image, target) in enumerate(retain_loader):
            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean, _ = model(image)
            loss = criterion(output_clean, target.to(torch.int64))

            optimizer.zero_grad()
            loss.backward()


            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]  # 这里乘

            optimizer.step()


            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Time {3:.2f}'.format(
                    epoch, i, loader_len, end - start, loss=losses, top1=top1))


    return top1.avg