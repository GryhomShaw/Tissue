import os
import json
import random
import imgaug as ia
import imgaug.augmenters as iaa

import cv2
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
import numpy as np

from eic_utils import cp
from lib.config import config


class MILdataset(data.Dataset):
    def __init__(self, data_path, transform=None):
        # Flatten grid
        grid = []  # path for per patch
        slideIDX = []
        slidenames = []
        targets = []
        slideLen = [0]
        idx = 0
        for each_file in data_path:
            slidenames.append(each_file.split('/')[-1])
            if 'pos' in each_file:
                targets.append(1)
            else:
                targets.append(0)
            slideLen.append(slideLen[-1] + len(os.listdir(each_file)))
            for each_patch in os.listdir(each_file):
                img_path = os.path.join(each_file, each_patch)
                grid.append(img_path)
                slideIDX.append(idx)
            idx += 1
            cp('(#g)index: {}(#)\t(#r)name: {}(#)\t(#y)len: {}(#)'.format(idx, each_file.split('/')[-1],
                                                                          len(os.listdir(each_file))))
        cp('(#g)total: {}(#)'.format(len(grid)))

        assert len(targets) == len(slidenames), print("targets and names not match")
        assert len(slideIDX) == len(grid), print("idx and mask not match")

        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Crop(percent=(0.1, 0.3)),
            # iaa.Sometimes(0.4, iaa.GaussianBlur(sigma=(0, 0.5))),
            # iaa.Sometimes(0.6, iaa.ContrastNormalization((0.75, 1.5))),
            # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Rot90((1, 3))
        ], random_order=True
        )

        self.slidenames = slidenames
        self.targets = targets
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.slideLen = slideLen  # patches for each slide
        self.size = config.DATASET.PATCHSIZE
        self.seq = seq
        self.multi_scale = config.DATASET.MULTISCALE
        self.unit = self.size
        self.overlap = config.DATASET.OVERLAP
        self.step = self.unit
        self.blocks_per_slide = 1
        self.ms_slideLen = self.slideLen  #for multi_scale
        self.ms_slideIDX = self.slideIDX

    def setmode(self, mode):

        self.mode = mode
        self.unit = self.size // self.multi_scale[mode]
        self.step = int(self.unit * (1-self.overlap))
        self.blocks_per_slide = (self.size - self.unit) // self.step + 1
        self.ms_slideLen = (np.array(self.slideLen) * pow(self.blocks_per_slide, 2)).tolist()
        self.ms_slideIDX = []
        for slide_idx in np.arange(1, len(self.ms_slideLen)):
            self.ms_slideIDX.extend([slide_idx - 1] * (self.ms_slideLen[slide_idx] - self.ms_slideLen[slide_idx - 1]))

    def maketraindata(self, idxs):
        self.t_data = []
        for each in idxs:
            cur_idx = each[0]
            cur_scale = self.multi_scale[each[1]]
            cur_unit, cur_blocks_per_slide = self.calc_scaleInfo(cur_scale)
            img_idx = int(cur_idx // pow(cur_blocks_per_slide, 2))
            part_idx = int(cur_idx % pow(cur_blocks_per_slide, 2))
            self.t_data.append([self.grid[img_idx], self.targets[self.slideIDX[img_idx]], part_idx, cur_unit,
                                cur_blocks_per_slide])


    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def get_part(self, idx, blocks_per_slide, step): #根据part_id 转换为行列坐标

        row_idx = idx // blocks_per_slide
        col_idx = idx % blocks_per_slide
        return int(row_idx * step), int(col_idx * step)

    def calc_scaleInfo(self, scale):
        cur_unit = self.size // scale
        cur_step = cur_unit * (1 - self.overlap)
        cur_blocks_per_slide = (self.size - cur_unit) // cur_step + 1
        return int(cur_unit), int(cur_blocks_per_slide)

    def __getitem__(self, index):

        if self.mode == -1:# train
            img_path, target, part_idx, cur_unit, cur_blocks_per_slide = self.t_data[index]
            row, col = self.get_part(part_idx, cur_blocks_per_slide, int(cur_unit*(1-self.overlap)))
            img = cv2.imread(img_path)[:, :, ::-1]
            img = img[row:row+cur_unit, col:col+cur_unit, :]

            if cur_unit != 224:
                ifaug = np.random.randint(2)
                if ifaug == 1:  # 50%
                    img = self.seq.augment_image(img)
                img = cv2.resize(img, (224, 224))
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        else:
            img_idx = index // pow(self.blocks_per_slide, 2)
            part_idx = index % pow(self.blocks_per_slide, 2)
            row, col = self.get_part(part_idx, self.blocks_per_slide, self.step)
            img_path = self.grid[img_idx]
            img = cv2.imread(img_path)[:, :, ::-1]
            img = img[row:row+self.unit, col:col + self.unit, :]
           # print(img.shape)
            if img is None:
                print(img_path, img.shape, row, col, row+self.unit, col+self.unit)
            #print(img.shape, row, col, self.unit)
            if self.unit != 224:
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
            if self.transform is not None:
                img = self.transform(img)
            return img, img_idx, row, col

    def __len__(self):
        if self.mode == -1:
            return len(self.t_data)
        else:
            return len(self.grid) * pow(self.blocks_per_slide, 2)


if __name__ == '__main__':
    data_path = ['/home/gryhomshaw/SSD1T/xiaoguohong/MIL_Tissue/patch/pos/1800127001_2019-04-30 10_40_02-lv1-1001-22444-3989-3967']
    dset = MILdataset(data_path)
    dset.setmode(0)

    cv2.imwrite("temp.jpg", dset[0][0])
    dset.setmode(2)
    for i in range(7):
        cv2.imwrite('temp_{}.jpg'.format(str(i)), dset[i][0])