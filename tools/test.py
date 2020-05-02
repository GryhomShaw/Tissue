import sys
import os
import numpy as np
import argparse
import random
import bisect
import PIL
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models

import shutil
import json
import cv2
from eic_utils import cp, procedure
from sklearn import mixture
from sklearn.metrics import f1_score

from lib.config import config
from lib.config import update_config
from lib.models.model_select import get_model
from lib.dataloader.dataset import MILdataset
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS


def get_args():
    parser = argparse.ArgumentParser(description='Tissue_test')
    parser.add_argument('--cfg', type=str, help='path of config file')
    parser.add_argument('opts', type=None, nargs=argparse.REMAINDER, help='modify some configs')
    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = get_args()
    #load model
    with procedure('load model'):
        model = get_model(config)
        model.fc = nn.Linear(model.fc.in_features, config.NUMCLASSES)
        model = nn.DataParallel(model.cuda())
        if config.TEST.RESUME:
            ch = torch.load(config.TEST.CHECKPOINT)
            model.load_state_dict(ch['state_dict'])

        cudnn.benchmark = True

    with procedure('prepare dataset'):
        # normalization
        normalize = transforms.Normalize(mean=config.DATASET.MEAN, std=config.DATASET.STD)
        trans = transforms.Compose([transforms.ToTensor(), normalize])
        #load data
        with open(config.DATASET.SPLIT) as f:
            data = json.load(f)

        dset = MILdataset(data['val_pos'] + data['val_neg'],  trans)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=config.TEST.BATCHSIZE, shuffle=False,
            num_workers=config.WORKERS, pin_memory=True)

    #dset.setmode(len(config.DATASET.MULTISCALE)-1)
    dset.setmode(len(config.DATASET.MULTISCALE)-2)
    probs, img_idxs, rows, cols = inference(loader, model)
    patch_info = probs_parser(probs, img_idxs, rows, cols, dset, config.DATASET.MULTISCALE[-2])
    #print(patch_info)
    for img_path, labels in patch_info.items():
        if len(labels) == 0:
            continue
        plot_label(img_path, labels, config.DATASET.MULTISCALE[-2])

    #maxs = group_max(dset.slideLen, probs[:, 1], len(dset.targets), config.DATASET.MULTISCALE[0])


def probs_parser(probs, img_idxs, rows, cols, dset, scale):
    assert isinstance(probs, np.ndarray) and isinstance(img_idxs, np.ndarray) and isinstance(rows, np.ndarray) and \
           isinstance(cols, np.ndarray), print("TYPE ERROR")

    assert probs.shape[0] == img_idxs.shape[0] and img_idxs.shape[0] == rows.shape[0] and \
           rows.shape[0] == cols.shape[0], print("LENGTH ERROR")

    prefix = 'tissue-train-'
    slide_len = np.array(dset.slideLen) * pow(scale, 2)
    assert slide_len[-1] == probs.shape[0], print("VAL ERROR")
    slide_names = [prefix + dset.grid[each_idx].split('/')[-3] + '/' + dset.grid[each_idx].split('/')[-2]+'.jpg'
                    for each_idx in img_idxs]
    row_offsets = np.array([int(dset.grid[each_idx].split('/')[-1].split('_')[0]) for each_idx in img_idxs])
    col_offsets = np.array([int((dset.grid[each_idx].split('/')[-1].split('_')[-1]).replace('.jpg', ''))
                            for each_idx in img_idxs])
    img_path = [os.path.join(os.path.join(config.DATASET.ROOT, each_name)) for each_name in slide_names]
    rows = rows + row_offsets
    cols = cols + col_offsets
    res = {}
    for idx in range(slide_len.shape[0]-1):
        start = slide_len[idx]
        end = slide_len[idx+1]
        res[img_path[start]] = []
        for label_idx in range(start, end):
            if probs[label_idx][0] > probs[label_idx][1]:
                continue
            res[img_path[start]].append([rows[label_idx], cols[label_idx], probs[label_idx][1]])
    return res


def plot_label(img_path, labels, scale):
    slide_class = img_path.split('/')[-2].split('-')[-1]
    slide_name = img_path.split('/')[-1] #pos/XXXX
    if not os.path.isdir(os.path.join(config.TEST.OUTPUT, slide_class)):
        os.makedirs(os.path.join(config.TEST.OUTPUT, slide_class))
    save_path = os.path.join(os.path.join(config.TEST.OUTPUT, slide_class), slide_name)
    img = cv2.imread(img_path)
    print(img.shape)
    patch_size = config.DATASET.PATCHSIZE // scale
    for label in labels:
        h, w = label[0], label[1]
        start = (w, h)
        end = (w+patch_size, h+patch_size)
        cv2.rectangle(img, start, end, (0, 255, 0), 5)
        cv2.putText(img, str(round(label[2], 2)), (w+25, h+25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    cv2.imwrite(save_path, img)


def inference(loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset), 2)
    img_idxs = []
    rows = []
    cols = []
    with torch.no_grad():
        for i, (input, img_idx, row, col) in enumerate(loader):
            print('Batch: [{}/{}]'.format(i+1, len(loader)))
            input = input.cuda()
            img_idxs.extend(list(img_idx.numpy()))
            rows.extend(list(row.numpy()))
            cols.extend(list(col.numpy()))
            output = F.softmax(model(input), dim=1)
            probs[i*config.TEST.BATCHSIZE:i*config.TEST.BATCHSIZE+input.size(0), :] = output.detach().clone()
    return probs.cpu().numpy(), np.array(img_idxs), np.array(rows), np.array(cols)


def group_max(slideLen, data, nmax, scale):
    groups = []
    slideLen = np.array(slideLen[:]) * pow(scale, 2)
    for slide_idx in np.arange(1, len(slideLen)):
        groups.extend([slide_idx-1] * (slideLen[slide_idx] - slideLen[slide_idx-1]))
    groups = np.array(groups)
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


def calc_err(pred, real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    f1 = f1_score(real, pred, average='binary')
    return err, fpr, fnr, f1


if __name__ == '__main__':
    main()

