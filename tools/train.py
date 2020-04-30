import os
import sys
import time
import numpy as np
import argparse
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.models as models
from eic_utils import procedure, cp

from lib.utils.summary import TensorboardSummary
from lib.utils.average import AverageMeter, ProgressMeter
from sklearn.metrics import f1_score

from lib.config import config
from lib.config import update_config
from lib.dataloader.dataset import MILdataset
from lib.models.model_select import get_model
from lib.loss.criterion import FocalLoss
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='ECDP_NCIC')
    parser.add_argument('--cfg', type=str, help='path of config name')
    parser.add_argument('opts', type=None, nargs=argparse.REMAINDER, help='modify some default cfgs')
    args = parser.parse_args()
    update_config(config, args)
    return parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] =config.GPUS
best_acc = 0.

def main():
    args = get_args()
    global best_acc
    #cnn
    with procedure('init model'):
        model = get_model(config)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = torch.nn.parallel.DataParallel(model.cuda())

    with procedure('loss and optimizer'):
        criterion = FocalLoss(config.TRAIN.LOSS.GAMMA).cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.LR)

    start_epoch = 0

    if config.TRAIN.RESUME:
        with procedure('resume model'):
            start_epoch, best_acc, model, optimizer = load_model(model, optimizer)

    cudnn.benchmark = True
    #normalization
    normalize = transforms.Normalize(mean=config.DATASET.MEAN, std=config.DATASET.STD)
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    with procedure('prepare dataset'):
        #load data
        data_split = config.DATASET.SPLIT
        with open(data_split) as f:
            data = json.load(f)

        train_dset = MILdataset(data['train_neg'] + data['train_pos'], trans)
        print(len(train_dset.slideLen))
        train_loader = DataLoader(
            train_dset,
            batch_size=config.TRAIN.BATCHSIZE, shuffle=False,
            num_workers=config.WORKERS, pin_memory=True)
        if config.TRAIN.VAL:
            val_dset = MILdataset(data['val_pos']+data['val_neg'], trans)
            val_loader = DataLoader(
                val_dset,
                batch_size=config.TRAIN.BATCHSIZE, shuffle=False,
                num_workers=config.WORKERS, pin_memory=True)

    with procedure('init tensorboardX'):
        train_log_path = os.path.join(config.TRAIN.OUTPUT, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        if not os.path.isdir(train_log_path):
            os.makedirs(train_log_path)
        tensorboard_path = os.path.join(train_log_path, 'tensorboard')
        if not os.path.isdir(tensorboard_path):
            os.makedirs(tensorboard_path)
        summary = TensorboardSummary(tensorboard_path)
        writer = summary.create_writer()

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        index = []
        for idx, each_scale in enumerate(config.DATASET.MULTISCALE):
            train_dset.setmode(idx)
            probs = inference(epoch, train_loader, model)
            topk = group_argtopk(probs, train_dset.targets, train_dset.slideLen, each_scale)
            index.extend([[each[0], each[1]] for each in zip(topk, [idx]*len(topk))])
        train_dset.maketraindata(index)
        train_dset.shuffletraindata()
        train_dset.setmode(-1)
        loss = train(epoch, train_loader, model, criterion, optimizer, writer)
        cp('(#r)Training(#)\t(#b)Epoch: [{}/{}](#)\t(#g)Loss:{}(#)'.format(epoch+1, config.TRAIN.EPOCHS, loss))

        if config.TRAIN.VAL and (epoch+1) % config.TRAIN.VALGAP == 0:
            val_dset.setmode(0)
            probs = inference(epoch, val_loader, model)
            maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
            threshold = 0.5
            pred = [1 if x >= threshold else 0 for x in maxs]
            err, fpr, fnr, f1 = calc_err(pred, val_dset.targets)
            cp('(#y)Vaildation\t(#)(#b)Epoch: [{}/{}]\t(#)(#g)Error: {}\tFPR: {}\tFNR: {}\tF1: {}(#)'.format(epoch+1, config.TRAIN.EPOCHS, err, fpr, fnr, f1))
            writer.add_scalar('Val/err', err, epoch)
            writer.add_scalar('Val/f1', f1, epoch)

            if 1-err >= best_acc:
                best_acc = 1-err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer':optimizer.state_dict()
                }
                torch.save(obj, os.path.join(train_log_path, 'BestCheckpoint.pth'))


def inference(run, loader, model):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(len(loader), [batch_time, data_time],
                             prefix=cp.trans("(#b)[INF](#) Epoch: [{}]".format(run)))
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        end = time.time()
        for i, input in enumerate(loader):
            data_time.update(time.time() - end)
            input = input.cuda()
            batch_time.update(time.time()-end)
            output = F.softmax(model(input), dim=1)
            probs[i*config.TRAIN.BATCHSIZE:i*config.TRAIN.BATCHSIZE+input.size(0)] = output.detach()[:, 1].clone()
            if (i+1) % 100 == 0:
                progress.display(i)
            end = time.time()
    return probs.cpu().numpy()


def train(run, loader, model, criterion, optimizer, writer):
    losses = AverageMeter('Loss', ':.4f')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(len(loader), [batch_time, data_time, losses],
                             prefix=cp.trans("(#b)[TRN](#) Epoch: [{}]".format(run)))
    model.train()
    running_loss = 0.
    end = time.time()
    criterion.adjust_gamma(gamma_ploy(run))
    for i, (input, target) in enumerate(loader):
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        losses.update(loss.item(),input.size(0))
        running_loss += loss.item()*input.size(0)
        progress.display(i)
        writer.add_scalar('train/loss', loss.item(), run * len(loader) + i)
        end =time.time()
    return running_loss/len(loader.dataset)


def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    f1 = f1_score(real, pred, average='binary')
    return err, fpr, fnr, f1


def group_argtopk(data, targets, slideLen, scale):
    k = config.TRAIN.SELECTNUM * scale
    slideLen = np.array(slideLen[:]) * pow(scale, 2)
    groups = []
    for slide_idx in np.arange(1, len(slideLen)):
        groups.extend([slide_idx-1] * (slideLen[slide_idx] - slideLen[slide_idx-1]))
    groups = np.array(groups)
    assert groups.shape[0] == slideLen[-1], print("SHAPE ERROR")
    assert groups.shape[0] == data.shape[0], print("SHAPE ERROR")
    order = np.lexsort((data, groups))
    groups = groups[order]
    index = np.full(len(groups), False)
    if config.TRAIN.MODE == 'max-max': # max-max
        index[-k:] = True
        index[:-k] = groups[k:] != groups[:-k]
    else:
        for idx in range(1, slideLen.shape[0]):
            cur_id = idx-1
            if targets[cur_id] == 1:
                index[slideLen[idx] - k:slideLen[idx]] = True
            else:
                index[slideLen[cur_id]:slideLen[cur_id] + k] = True
    return list(order[index])


def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out


def load_model(model, optimizer):
    ckpt = torch.load(config.TRAIN.CHECKPOINT)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch'], ckpt['best_acc'], model, optimizer


def gamma_ploy(epoch):
    if epoch < 2:
        return 0
    else:
        return 1


if __name__ == '__main__':
    main()

