import sys
import os
import time
import numpy as np
import argparse
import random
import bisect
import PIL
import multiprocessing
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
from lib.core.function import inference_vt, inference_cam
from lib.utils.parser import probs_parser, group_max, get_mask
from lib.core.criterion import calc_err
from lib.core.criterion import calc_dsc
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.GPUS))


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
        model = nn.DataParallel(model.cuda())
        if config.TEST.RESUME:
            ch = torch.load(config.TEST.CHECKPOINT)
            model.load_state_dict(ch['state_dict'])
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
        data_list = [os.path.join(data_root, each_slide) for each_slide in os.listdir(data_root)][:10]
        dset = MILdataset(data_list,  trans)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=config.TEST.BATCHSIZE, shuffle=False,
            num_workers=config.WORKERS, pin_memory=True)
    time_fromat = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    output_path = os.path.join(os.path.join(config.TEST.OUTPUT, config.MODEL), config.TRAIN.MODE + '_' + time_fromat)
    patch_info = {}

    for idx, each_scale in enumerate(config.DATASET.MULTISCALE):
        dset.setmode(idx)
        probs, img_idxs, rows, cols = inference_vt(0, loader, model)
        if config.TEST.CAM:
            inference_cam(0, loader, model)
        parser_end = time.time()
        res = probs_parser(probs, img_idxs, rows, cols, dset, each_scale)
        print("finish parser: {}".format(time.time() - parser_end))
        merage_end = time.time()
        for key, values in res.items():
            if key not in patch_info:
                patch_info[key] = values
            else:
                patch_info[key].extend(values)
        print("finish merage: {}".format(time.time() - merage_end))
        '''
        for img_path, labels in res.items():
            if len(labels) == 0:
                continue
            plot_label(img_path, labels, each_scale, output_path)
        '''
    masks_end = time.time()
    res = []
    dsc = []
    with multiprocessing.Pool(processes=16) as pool:
        for each_img, each_labels in patch_info.items():
            res.append(pool.apply(get_mask, (each_img, each_labels, output_path)))
    pool.join()
    for each_res in res:
        print(type(each_res), each_res)
        dsc.extend([each_val for each_val in each_res.values()])

    print("finish get mask: {}".format(time.time() - masks_end))

    mean_dsc = np.array(dsc).mean()
    print(mean_dsc, dsc)


def plot_label(img_path, labels, scale, output_path):
    slide_class = img_path.split('/')[-2].split('-')[-1]
    slide_name = img_path.split('/')[-1] .replace('.jpg', '')
    if not os.path.isdir(os.path.join(output_path, slide_class)):
        os.makedirs(os.path.join(output_path, slide_class))
    save_path = os.path.join(os.path.join(output_path, slide_class), slide_name+'(' + str(scale) + ').jpg')
    img = cv2.imread(img_path)
    patch_size = config.DATASET.PATCHSIZE // scale
    for label in labels:
        if labels[2][0] > labels[2][1]:
            continue
        h, w = label[0], label[1]
        start = (w, h)
        end = (w+patch_size, h+patch_size)
        cv2.rectangle(img, start, end, (0, 255, 0), 5)
        cv2.putText(img, str(round(label[2][1], 2)), (w+25, h+25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(save_path, img)


def save_img(img, save_path, slide_name):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, slide_name), img)


if __name__ == '__main__':
    main()

