import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
from os.path import join
import math
import random
import time
import net_sphere
import numpy as np
import glob
import shutil
import pandas as pd
from PIL import Image
import mydataset
import builder
import util.util as util
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score                          



def run(args, save_dir):
    best_acc = 0
    
    net = getattr(builder, args.arch)()

    # BCE loss, Adam opt
    criterion = {}

    criterion['BCE'] = nn.BCEWithLogitsLoss().cuda('cuda')

    parameters = list(filter(lambda p: p.requires_grad, net.parameters()))

    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = getattr(mydataset, args.data)(train='train', transform=transforms.Compose([
        transforms.Resize((args.psize, args.psize)),
        transforms.ToTensor(),
        normalize
    ]), split=args.split)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=32, pin_memory=True)

    val_dataset = getattr(mydataset, args.data)(train='val', transform=transforms.Compose([
            transforms.Resize((args.psize, args.psize)),
            transforms.ToTensor(),
            normalize
        ]), split=args.split)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=32, pin_memory=True)

    test_dataset = getattr(mydataset, args.data)(train='test', transform=transforms.Compose([
            transforms.Resize((args.psize, args.psize)),
            transforms.ToTensor(),
            normalize
        ]), split=args.split)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False,
        num_workers=32, pin_memory=True)

    if args.evaluate:
        print('load model from: ', join(save_dir, args.pretrained))
        checkpoint = torch.load(join(save_dir, args.pretrained), map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        msg = net.load_state_dict(state_dict, strict=True)
        print(msg.missing_keys)
        net.cuda()
        
        writer = SummaryWriter(os.path.join(save_dir, 'test'))
        _ = validate(val_loader, net, 0, criterion, args, writer, 'val')
        best_metrics = validate(test_loader, net, 0, criterion, args, writer, 'test')
        print('****best testing result: epoch: {}, acc: {}, auc: {}, sen: {}, spe: {}, loss: {}'.format(0, best_metrics[0], best_metrics[1], \
    best_metrics[2], best_metrics[3], best_metrics[4]))
        return

    net.cuda()
    writer = SummaryWriter(os.path.join(save_dir, 'log'))
    # EarlyStopping = util.EarlyStopping(save_dir=save_dir, args=args)
    # monitor_values = {'acc':0, 'auc':1, 'loss':4}
    # monitor_idx = monitor_values[args.monitor]
    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, args.lr, epoch, args)

        # train for one epoch
        train(train_loader, net, criterion, optimizer, epoch, args, writer)

        # evaluate on validation set
        metrics = validate(val_loader, net, epoch, criterion, args, writer, 'val')


        # save the best validation performance model
        if epoch > 8:
            is_best = metrics[0][0] >= best_acc
            best_acc = max(metrics[0][0], best_acc)

        if is_best:
            # evaluate on testing set
            best_metrics = validate(val_loader, net, epoch, criterion, args, writer, 'test')
            best_epoch = epoch
        else:
            _ = validate(val_loader, net, epoch, criterion, args, writer, 'test')


    print('****best testing result: epoch: {}, acc: {}, auc: {}, sen: {}, spe: {}, loss: {}'.format(best_epoch, best_metrics[0], best_metrics[1], \
    best_metrics[2], best_metrics[3], best_metrics[4]))

def train(train_loader, model, criterion,  optimizer, epoch, args, writer):
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Acc', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [losses, accs],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda().float()

        output = model(images).view(-1).float()
        if i == 0:
            outputs = output
            targets = target
        else:
            outputs = torch.cat((outputs, output), 0)
            targets = torch.cat((targets, target), 0)

        loss = criterion['BCE'](output, target)

        acc, _, _ = accuracy(output, target, args, False)
        
        losses.update(loss.item(), images.size(0))
        accs.update(acc[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.freq == 0:
            progress.display(i)

    acc, sen, spe = accuracy(outputs, targets, args, False)
    writer.add_scalar("Loss/train", losses.avg, epoch)
    writer.add_scalar("Accuracy/train", accs.avg, epoch)
    writer.add_scalar("sen/train", sen, epoch)
    writer.add_scalar("spe/train", spe, epoch)


def validate(val_loader, model, epoch, criterion, args, writer, val='val'):
    losses = AverageMeter('Loss', ':.4e')

    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):    
            images = images.cuda()
            target = target.cuda().float()

            output = model(images).view(-1).float()
            if i == 0:
                outputs = output
                targets = target
            else:
                outputs = torch.cat((outputs, output), 0)
                targets = torch.cat((targets, target), 0)
            loss = criterion['BCE'](output, target)
            losses.update(loss.item(), images.size(0))

    acc, sen, spe, auc = accuracy(outputs, targets, args)

    if val == 'val':
        print(' **Validation Acc {acc:.3f} sen {sen:.3f} spe {spe:.3f} AUC {auc:.3f}'
            .format(acc=acc[0], sen=sen, spe=spe, auc=auc))
    else:
        print(' ***Testing Acc {acc:.3f} sen {sen:.3f} spe {spe:.3f} AUC {auc:.3f}'
            .format(acc=acc[0], sen=sen, spe=spe, auc=auc))
    writer.add_scalar("Loss/"+val, losses.avg, epoch)
    writer.add_scalar("Accuracy/"+val, acc, epoch)
    writer.add_scalar("sen/"+val, sen, epoch)
    writer.add_scalar("spe/"+val, spe, epoch)
    writer.add_scalar("auc/"+val, auc, epoch)

    return acc, auc, sen, spe, losses.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, args, test=True):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        # _, pred = output.topk(maxk, 1, True, True)
        scores = torch.sigmoid(output).cpu().numpy()
        pred = (torch.sigmoid(output) > args.threshold).float().view(-1, 1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1))

        if test:
            auc = roc_auc_score(target.view(-1).cpu().numpy(), scores)

        tp = torch.sum(torch.logical_and(correct, pred == 1)).float()
        tn = torch.sum(torch.logical_and(correct, pred == 0)).float()

        res = []

        correct = correct.reshape(-1).float().sum(0, keepdim=True)
        acc = correct.mul_(100.0 / batch_size)
        res.append(acc.view(-1).cpu().numpy())
        # if torch.sum(target == 1).float() == 0:
        #     recall = 0
        # else:
        recall = tp / torch.sum(target == 1).float()

        # if torch.sum(target == 0).float() == 0:
        #     spe = 0
        # else:
        spe = tn / torch.sum(target == 0).float()

        res.append(recall*100)
        res.append(spe*100)
        if test:
            res.append(auc*100)
        return res

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DI transfer learning')
    parser.add_argument('--model', default='', type=str,
                        help='path to the model')
    parser.add_argument('--arch', default='resnet18', type=str,
                        help='architecture')
    parser.add_argument('--data', default='DIPn', type=str,
                        help='dataset')                    
    parser.add_argument('--split', default=42, type=int,
                        help='split random seed')
    parser.add_argument('-b', '--batch_size', default=60, type=int,
                        help='batch size')          
    parser.add_argument('--epochs', default=20, type=int,
                        help='number of epochs') 
    parser.add_argument('-p', '--psize', default=224, type=int,
                        help='input size')                      
    parser.add_argument('--code', default='test', type=str,
                        help='exp code')                        
    parser.add_argument('-t', '--threshold', default=0.5, type=float,
                        help='accuracy threshold') 
    parser.add_argument('--lr', default=0.00001, type=float, 
                        help='init learning rate')          
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')  
    parser.add_argument('--pretrained', default='model_best.pth.tar', type=str, 
                        help='pretrained model for validate') 
    parser.add_argument('--freq', default=50, type=int, 
                        help='training log display frequency')    

    # parser.add_argument('--patience', default=5, type=int, 
                        # help='early stopping patience') 
    # parser.add_argument('--stop_epoch', default=8, type=int, 
                        # help='start epoch to activate early stopping counting') 
    # parser.add_argument('--monitor', default='auc', type=str, 
                        # help='value to monitor for early stopping')  

    # parser.add_argument('--l1l', default=False, type=bool, 
                        # help='use L1 loss')
    # parser.add_argument('--alpha', default=0.01, type=float, 
                        # help='L1 loss weight')
    # parser.add_argument('--sd', default=False, type=bool, 
                        # help='use spectral decoupling')
    # parser.add_argument('--beta', default=0.01, type=float, 
                        # help='spectral decoupling weight')                                                                            

    args = parser.parse_args()
    save_folder = './runs/' + args.code
    if not os.path.exists(save_folder):
        os.mkdir(save_folder) 

    for split in range(42, 47):
        print('===================================FOLD: ', split)
        args.split = split
        save_dir = save_folder + '/' + str(args.split)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)  
        run(args, save_dir)