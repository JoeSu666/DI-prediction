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
import numpy as np
import glob
import shutil
import pandas as pd
from PIL import Image
import mydataset
import builder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
                       

def run(args, save_dir):
    
    net = getattr(builder, args.arch)()

    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_dataset = getattr(mydataset, args.data)(train='test', transform=transforms.Compose([
            transforms.Resize((args.psize, args.psize)),
            transforms.ToTensor(),
            normalize
        ]), split=args.split)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False,
        num_workers=32, pin_memory=True)


    print('load model from: ', join(save_dir, args.pretrained))
    checkpoint = torch.load(join(save_dir, args.pretrained), map_location="cpu")

    state_dict = checkpoint['state_dict']
    msg = net.load_state_dict(state_dict, strict=True)
    print(msg.missing_keys)
    net.cuda()
    
    # _ = validate(val_loader, net, 0, criterion, args, writer, 'val')
    best_metrics = validate(test_loader, net, args, 'test')

    return best_metrics


def validate(val_loader, model, args, val='val'):

    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):    
            images = images.cuda()
            target = target.cuda().float()

            output = model(images)
            output = output.view(-1).float()

            if i == 0:
                outputs = output
                targets = target
            else:
                outputs = torch.cat((outputs, output), 0)
                targets = torch.cat((targets, target), 0)

    acc, sen, spe, auc = accuracy(outputs, targets, args)
    scores = torch.sigmoid(outputs).cpu().numpy()

    if val == 'val':
        print(' **Validation Acc {acc:.3f} sen {sen:.3f} spe {spe:.3f} AUC {auc:.3f}'
            .format(acc=acc, sen=sen, spe=spe, auc=auc))
    else:
        print(' ***Testing Acc {acc:.3f} sen {sen:.3f} spe {spe:.3f} AUC {auc:.3f}'
            .format(acc=acc, sen=sen, spe=spe, auc=auc))

    return acc, auc, sen, spe, scores



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
        scores = torch.sigmoid(output).cpu()
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
        res.append(acc.view(-1).cpu().numpy()[0])
        # if torch.sum(target == 1).float() == 0:
        #     recall = 0
        # else:
        recall = tp / torch.sum(target == 1).float()
        recall = recall.view(-1).cpu().numpy()[0]

        # if torch.sum(target == 0).float() == 0:
        #     spe = 0
        # else:
        spe = tn / torch.sum(target == 0).float()
        spe = spe.view(-1).cpu().numpy()[0]

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


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DI transfer learning')
    parser.add_argument('--model', default='', type=str,
                        help='path to the model')
    parser.add_argument('--arch', default='sphere20a', type=str,
                        help='architecture')
    parser.add_argument('--data', default='DIf', type=str,
                        help='dataset')                    
    parser.add_argument('--split', default=42, type=int,
                        help='split random seed')
    parser.add_argument('-b', '--batch_size', default=20, type=int,
                        help='batch size')          
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of epochs') 
    parser.add_argument('-p', '--psize', default=100, type=int,
                        help='input size')                      
    parser.add_argument('--code', default='test', type=str,
                        help='exp code')                        
    parser.add_argument('-t', '--threshold', default=0.5, type=float,
                        help='accuracy threshold')                                                                 

    parser.add_argument('--pretrained', default='model_best.pth.tar', type=str, 
                        help='pretrained model for validate') 
    parser.add_argument('--freq', default=50, type=int, 
                        help='training log display frequency')                                                                                      


    args = parser.parse_args()
    save_folder = './runs/'+args.code
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)   

    accs, aucs, sens, spes, scores = [], [], [], [], {}


    for i in [42]:
        args.split = i
        save_dir = join(save_folder, str(args.split))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir) 

        print('split:', args.split)
        acc, auc, sen, spe, score = run(args, save_dir)
        accs.append(acc)
        aucs.append(auc)
        sens.append(sen)
        spes.append(spe)
        scores[args.split] = score

    print('Testing AUC {:.3f} +- {:.3f}, ACC {:.3f} +- {:.3f}, SPE {:.3f} +- {:.3f}, SEN {:.3f} +- {:.3f}'
        .format(np.mean(aucs), np.std(aucs), np.mean(accs), np.std(accs), np.mean(spes), np.std(spes), np.mean(sens), np.std(sens)))
    df = pd.DataFrame(scores)
    df.to_csv(join(save_folder, 'results.csv'))