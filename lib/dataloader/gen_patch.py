import os
import cv2
import shutil
import argparse
import json
import numpy as np
import threadpool

from lib.config import config
from lib.config import update_config


def get_args():
    parser = argparse.ArgumentParser(description='MIL_TISSUE')
    parser.add_argument('--cfg', type=str, help='experiment configure file name')
    parser.add_argument('opts', type=None, nargs=argparse.REMAINDER, help='modify the options using command-line') \
        #将命令行剩下所有的参数封装为list
    args = parser.parse_args()
    update_config(config, args)
    return args


def cur_img(p):
    img_path, ispos = p
    psize = config.DATASET.PATCHSIZE
    step = config.DATASET.PATCHSTEP
    img_extension = os.path.splitext(img_path)[-1]
    threshold = config.DATASET.PATCHTHRESH
    img_name = os.path.splitext(img_path.split('/')[-1])[0]
    output_path = os.path.join(os.path.join(config.DATASET.PATCH, 'pos' if ispos else 'neg'), img_name)
    mask_output_path = os.path.join(os.path.join(config.DATASET.PATCH, 'mask'))
    color_output_path = os.path.join(os.path.join(config.DATASET.PATCH, 'color'))


    try:
        os.makedirs(output_path)
    except:
        pass

    try:
        os.makedirs(mask_output_path)
    except:
        pass

    try:
        os.makedirs(color_output_path)
    except:
        pass
    img = cv2.imread(img_path)
    h, w = img.shape[0], img.shape[1]
    mask = ostu(img)
    cv2.imwrite(os.path.join(mask_output_path, img_name+'_mask.jpg'), mask)
    labels = []
    for i in range(0, h, step):
        for j in range(0, w, step):
            x2 = min(h, i + psize)
            y2 = min(w, j + psize)
            x1 = max(0, min(i, x2 - psize))
            y1 = max(0, min(j, y2 - psize))
            cur = img[x1:x2, y1:y2, :]
            cur_mask = mask[x1:x2, y1:y2]
            if np.sum(cur_mask) // 255 < threshold * psize * psize:
                continue
            labels.append([x1, y1, x2, y2])
            cur_img_name = "{}_{}.jpg".format(str(x1), str(y1))
            print(os.path.join(output_path, cur_img_name))
            cv2.imwrite(os.path.join(output_path, cur_img_name), cur)
    color_img = color(img, labels)
    cv2.imwrite(os.path.join(color_output_path, img_name+'_color.jpg'), color_img)





def check():
    patch_root = config.DATASET.PATCH
    count = {}
    for each in ['pos', 'neg']:
        count[each] = 0
        for each_dirs in os.listdir(os.path.join(patch_root, each)):
            cur_path = os.path.join(os.path.join(patch_root, each), each_dirs)
            if len(os.listdir(cur_path)) < config.DATASET.LOWWER or len(os.listdir(cur_path)) > config.DATASET.UPPER:
                shutil.rmtree(cur_path)
                count[each] += 1
                print("Remove {}".format(cur_path))
    print(count)


def color(img, coords):
    for each_coord in coords:
        img = cv2.rectangle(img, (each_coord[1], each_coord[0]), (each_coord[3], each_coord[2]), (255, 0, 0), 3)
    return img


if __name__ == '__main__':
    args = get_args()
    pos_path = config.DATASET.POS
    neg_path = config.DATASET.NEG

    params = []

    for roots, _, filenames in os.walk(pos_path):
        for each_file in filenames:
            if '_mask' not in each_file:
                params.append([os.path.join(roots, each_file), True])
    for roots, _, filenames in os.walk(neg_path):
        for each_file in filenames:
            params.append([os.path.join(roots, each_file), False])
    #print(len(params))
    pool = threadpool.ThreadPool(config.DATASET.POOLSIZE)
    requests = threadpool.makeRequests(cur_img, params)
    [pool.putRequest(req) for req in requests]
    pool.wait()

    #check()


