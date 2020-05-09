import os
import sys
import time
import numpy as np
import argparse
import random
import json
import cv2

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
from lib.utils.parser import group_max,  group_argtopk, probs_parser, get_mask

from lib.config import config
from lib.config import update_config
from lib.dataloader.dataset import MILdataset
from lib.models.model_select import get_model
from lib.core.criterion import FocalLoss
from lib.core.criterion import calc_err, calc_dsc
from lib.core.function import trainer
from lib.core.function import inference
from lib.core.function import inference_vt

from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='ECDP_NCIC')
    parser.add_argument('--cfg', type=str, help='path of config name')
    parser.add_argument('opts', type=None, nargs=argparse.REMAINDER, help='modify some default cfgs')
    args = parser.parse_args()
    update_config(config, args)
    return parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.GPUS))
best_dsc = 0.


def main():

    args = get_args()
    global best_dsc
    #cnn
    with procedure('init model'):
        model = get_model(config)
        model = torch.nn.parallel.DataParallel(model.cuda())

    with procedure('loss and optimizer'):
        criterion = FocalLoss(config.TRAIN.LOSS.GAMMA, config.DATASET.ALPHA).cuda()
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
        train_loader = DataLoader(
            train_dset,
            batch_size=config.TRAIN.BATCHSIZE, shuffle=False,
            num_workers=config.WORKERS, pin_memory=True)
        if config.TRAIN.VAL:
            val_dset = MILdataset(data['val_pos']+data['val_neg'], trans)
            val_loader = DataLoader(
                val_dset,
                batch_size=config.TEST.BATCHSIZE, shuffle=False,
                num_workers=config.WORKERS, pin_memory=True)

    with procedure('init tensorboardX'):
        train_log_path = os.path.join(config.TRAIN.OUTPUT, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        if not os.path.isdir(train_log_path):
            os.makedirs(train_log_path)
        tensorboard_path = os.path.join(train_log_path, 'tensorboard')
        with open(os.path.join(train_log_path, 'cfg.yaml'), 'w') as f:
            print(config, file=f)
        if not os.path.isdir(tensorboard_path):
            os.makedirs(tensorboard_path)
        summary = TensorboardSummary(tensorboard_path)
        writer = summary.create_writer()

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        index = []
        for idx, each_scale in enumerate(config.DATASET.MULTISCALE):
            train_dset.setmode(idx)
            #print(len(train_loader), len(train_dset))
            probs = inference(epoch, train_loader, model)
            topk = group_argtopk(train_dset.ms_slideIDX[:], probs, train_dset.targets[:], train_dset.ms_slideLen[:],
                                 each_scale)
            index.extend([[each[0], each[1]] for each in zip(topk, [idx]*len(topk))])
        train_dset.maketraindata(index)
        train_dset.shuffletraindata()
        train_dset.setmode(-1)
        loss = trainer(epoch, train_loader, model, criterion, optimizer, writer)
        cp('(#r)Training(#)\t(#b)Epoch: [{}/{}](#)\t(#g)Loss:{}(#)'.format(epoch+1, config.TRAIN.EPOCHS, loss))

        if config.TRAIN.VAL and (epoch+1) % config.TRAIN.VALGAP == 0:
            patch_info = {}
            for idx, each_scale in enumerate(config.DATASET.MULTISCALE):
                val_dset.setmode(idx)
                probs, img_idxs, rows, cols = inference_vt(epoch, val_loader, model)
                res = probs_parser(probs, img_idxs, rows, cols, val_dset, each_scale)

                for key, val in res.items():
                    if key not in patch_info:
                        patch_info[key] = val
                    else:
                        patch_info[key].extend(val)
            masks = get_mask(patch_info)
            dsc = []
            for img_path, pred in masks.items():
                mask_path = img_path.replace('.jpg', '_mask.jpg')
                if os.path.isfile(mask_path):
                    mask = cv2.imread(mask_path, 0)
                    mask = mask.astype(np.int) if np.max(mask) == 1 else (mask // 255).astype(np.int)
                    dsc.append(calc_dsc(pred, mask))

            dsc = np.array(dsc).mean()
            '''
            maxs = group_max(np.array(val_dset.slideLen), probs, len(val_dset.targets), config.DATASET.MULTISCALE[-1])
            threshold = 0.5
            pred = [1 if x >= threshold else 0 for x in maxs]
            err, fpr, fnr, f1 = calc_err(pred, val_dset.targets)

            cp('(#y)Vaildation\t(#)(#b)Epoch: [{}/{}]\t(#)(#g)Error: {}\tFPR: {}\tFNR: {}\tF1: {}(#)'.format(epoch+1, config.TRAIN.EPOCHS, err, fpr, fnr, f1))
            '''
            cp('(#y)Vaildation\t(#)(#b)Epoch: [{}/{}]\t(#)(#g)DSC: {}(#)'.format(epoch+1, config.TRAIN.EPOCHS, dsc))
            writer.add_scalar('Val/dsc', dsc, epoch)
            if dsc >= best_dsc:
                best_dsc = dsc
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_dsc': best_dsc,
                    'optimizer':optimizer.state_dict()
                }
                torch.save(obj, os.path.join(train_log_path, 'BestCheckpoint.pth'))


def load_model(model, optimizer):
    ckpt = torch.load(config.TRAIN.CHECKPOINT)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch'], ckpt['best_dsc'], model, optimizer


if __name__ == '__main__':
    main()

