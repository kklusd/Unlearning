import logging
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .utils import save_config_file, accuracy, save_checkpoint, AverageMeter

class SupClassifier(object):
    def __init__(self, *args, **kwargs):
        self.device = torch.device("cuda:0")
        self.args = kwargs['args']
        self.model = kwargs['model'].to()
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.train_loader = kwargs['train_loader']
        self.val_loader = kwargs['val_loader']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'retraining.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def train(self):
        # save config file
        save_config_file(self.writer.log_dir, self.args)
        best_acc = 0
        n_iter = 0
        logging.info(f"Start Classifier training for {self.args.epoches} epochs.")
        #logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epoches):
            self.model.train()
            losses = AverageMeter()
            top1 = AverageMeter()
            for idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = self.model(images)
                loss = self.criterion(output, labels)
                bsz = labels.shape[0]
                losses.update(loss.item(), bsz)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1.update(acc1[0], bsz)
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                if n_iter % 5 == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', acc1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', acc5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                    print('Train:[{0}][{1}]\t'
                            'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch_counter, n_iter, loss=losses, top1=top1))
                n_iter += 1
            _, val_acc = self.validate()
            if val_acc > best_acc:
                best_acc = val_acc
            print(f"Epoch: {epoch_counter}\tLoss: {losses.avg}\tTop1 accuracy: {top1.avg}")
            print('best accuracy:{:.2f}'.format(best_acc))
            self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {losses.avg}\tTop1 accuracy: {top1.avg}")
        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epoches)
        save_checkpoint({
            'epoch': self.args.epoches,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

    def validate(self):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        with torch.no_grad():
            for idx, (images, labels) in enumerate(self.val_loader):
                images = images.float().cuda()
                labels = labels.cuda()
                bsz = labels.shape[0]
                output = self.model(images)
                loss = self.criterion(output, labels)
                acc1, acc5 = accuracy(output, labels)
                top1.update(acc1[0], bsz)
                losses.update(loss.item(), bsz)
                if idx % 10 == 0:
                    print('Test:[{0}/{1}]\t'
                            'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(idx, len(self.val_loader), loss=losses, top1=top1))
        return losses.avg, top1.avg