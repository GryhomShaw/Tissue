import sys
import os
import time
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
from lib.core.function import inference_vt
from lib.utils.parser import probs_parser, group_max, get_mask
from lib.core.criterion import calc_err
from lib.core.criterion import calc_dsc
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


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
        #model.fc = nn.Linear(model.fc.in_features, config.NUMCLASSES)
        model = model.cuda()
        if config.TEST.RESUME:
            ch = torch.load(config.TEST.CHECKPOINT)
            model_dict = {}
            for key, val in ch['state_dict'].items():
                model_dict[key[7:]] = val
            model.load_state_dict(model_dict)
            print(ch['best_dsc'], ch['epoch'])

        cudnn.benchmark = True

    with procedure('prepare dataset'):
        # normalization
        normalize = transforms.Normalize(mean=config.DATASET.MEAN, std=config.DATASET.STD)
        trans = transforms.Compose([transforms.ToTensor(), normalize])
        #load data
        with open(config.DATASET.SPLIT) as f:
            data = json.load(f)
        data_root = '/home/gryhomshaw/SSD1T/xiaoguohong/MIL_Tissue/patch/pos'
        data_list = [os.path.join(data_root, each_slide) for each_slide in os.listdir(data_root)]
        dset = MILdataset(data_list,  trans)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=1, shuffle=False,
            num_workers=config.WORKERS, pin_memory=True)
    time_fromat = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    output_path = os.path.join(os.path.join(config.TEST.OUTPUT, config.MODEL), config.TRAIN.MODE + '_' + time_fromat)


    for idx, each_scale in enumerate([1]):
        dset.setmode(idx)
        model.eval()
        for i, (input, _, _, _) in enumerate(loader):
            input = input.cuda()
            input.requires_grad = True
            print(input.requires_grad)
            output = model(input)
            loss = output[:, 1]
            model.zero_grad()
            loss.backward()
            prob = F.softmax(output, dim=1)
            prob = prob.squeeze()
            if prob[1] <= 0.5:
                continue
            cur_weights = input.grad.cpu().numpy()
            cur_weights = cur_weights.squeeze()
            cur_weights = cur_weights.transpose([1, 2, 0])
            print(cur_weights.shape)
            cur_weights = np.max(cur_weights, axis=2)
            cur_weights = np.clip(cur_weights*5000, 0, 255).astype(np.int)
            cur_weights =cur_weights.squeeze()
            cur_slide = dset.grid[i].split('/')[-2]
            cur_dir = os.path.join(output_path, cur_slide)
            if not os.path.isdir(os.path.join(cur_dir)):
                os.makedirs(cur_dir)
            cur_name = dset.grid[i].split('/')[-1]

            cv2.imwrite(os.path.join(cur_dir, cur_name), cur_weights)






def save_img(img, save_path, slide_name):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, slide_name), img)


if __name__ == '__main__':
    main()

