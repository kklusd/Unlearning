import logging
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .utils import save_config_file, accuracy, save_checkpoint, AverageMeter
import numpy as np

class SupClassifier(object):
    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.epochs = kwargs['epochs']
        self.lr = kwargs["lr"]
        self.model = kwargs['model'].to(self.device)
        self.train_loader = kwargs['train_loader']
        self.val_loader = kwargs['val_loader']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'retraining.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def train(self):
        best_acc = 0
        n_iter = 0
        logging.info(f"Start Classifier training for {self.epochs} epochs.")
        #logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        val_loss_min = np.Inf
        counter = 0
        for epoch_counter in range(self.epochs):
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            self.model.train()
            losses = AverageMeter()
            top1 = AverageMeter()
            if counter/10 ==1:
                counter = 0
                self.lr = self.lr*0.5
            for idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                output, _ = self.model(images)
                loss = self.criterion(output, labels)
                bsz = labels.shape[0]
                losses.update(loss.item(), bsz)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1.update(acc1[0], bsz)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                if n_iter % 5 == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', acc1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', acc5[0], global_step=n_iter)
                    print('Train:[{0}][{1}]\t'
                            'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch_counter, n_iter, loss=losses, top1=top1))
                n_iter += 1
            val_loss, val_acc = self.validate()
            if val_acc > best_acc:
                best_acc = val_acc
            if val_loss <= val_loss_min:
                val_loss_min = val_loss
                counter = 0
            else:
                counter += 1
            print(f"Epoch: {epoch_counter}\tLoss: {losses.avg}\tTop1 accuracy: {top1.avg}")
            print('best accuracy:{:.2f}'.format(best_acc))
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {losses.avg}\tTop1 accuracy: {top1.avg}")
            logging.debug(f"Epoch:{epoch_counter}\tBest_val_acc:{best_acc}")
        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.epochs)
        save_checkpoint({
            'epoch': self.epochs,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

    def validate(self):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        self.model.eval()
        for idx, (images, labels) in enumerate(self.val_loader):
            images = images.float().to(self.device)
            labels = labels.to(self.device)
            bsz = labels.shape[0]
            output, _ = self.model(images)
            loss = self.criterion(output, labels)
            acc1, acc5 = accuracy(output, labels)
            top1.update(acc1[0], bsz)
            losses.update(loss.item(), bsz)
            if idx % 10 == 0:
                print('Test:[{0}/{1}]\t'
                        'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(idx, len(self.val_loader), loss=losses, top1=top1))
        return losses.avg, top1.avg