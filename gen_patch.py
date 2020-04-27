import argparse
import os
import numpy as np
import random
import  sys
sys.path.append('../')
from config import cfg
from eic_utils import cp,procedure
import openslide as opsl
import threadpool
import pandas as pd
import math
import cv2
import json
import PIL
from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = 99999999999


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--size', type=int, default=20000, help="size of cut")
    parse.add_argument('--patch_size', type=int, default=1024, help="size of patch")
    parse.add_argument('--scale', type=int, default=4, help="scale of resize")
    parse.add_argument('--poolsize', type=int, default=16, help="size of pool")
    parse.add_argument('--threshold', type=float, default=0.6, help="size of patch")
    parse.add_argument('--nums', type=int, default=300, help="nums of patch")
    parse.add_argument('--bin', type=float, default=0.05, help="margin")
    parse.add_argument('-o','--output',type=str,default=cfg.patch_data, help = 'path of patch')
    return parse.parse_args()

def rename():
    for root, dirs, filenames in os.walk(cfg.data_append_path):
        for each_tiff in filenames:
            if '.tif' in each_tiff and '.enp' not in each_tiff :
                img_path = os.path.join(root, each_tiff)
                new_path = os.path.join(root, each_tiff.split('_')[0].replace(".tif", "")+(".tif"))

                if img_path == new_path:
                    continue
                os.rename(img_path, new_path)
                cp('(#r){}(#)\t(#g){}(#)'.format(img_path, new_path))

def work(args):
    img_path, size, scale, output_patch_path, patch_size, nums, bin, thresh = args
    '''
    img_path: path of tif  (e.g. ./data_append/1/1.tif)
    size: size of patch (from tiff to jpeg) (e.g. 20000)
    scale: scale (riff2jpeg)  (e.g. 4)
    output_patch_path: path of patch (e.g. ./Patch/pos/1)
    patch_size: during cut_image (2048)
    '''
    output_mask_path = img_path[:-4] + '_mask.jpg'
    slide = opsl.OpenSlide(img_path)
    [n, m] = slide.dimensions

    with procedure('Tiff2Jpeg\t{}'.format(img_path.split('/')[-2])):
        if not os.path.isfile(output_mask_path) :
            blocks_pre_col = math.ceil(m / size)
            blocks_pre_row = math.ceil(n / size)
            row_cache = []
            img_cache = []
            for i in range(blocks_pre_col):
                for j in range(blocks_pre_row):
                    x = i * size
                    y = j * size
                    height = min(x + size, m) - x
                    width = min(y + size, n) - y
                    img = np.array(slide.read_region((y, x), 0, (width, height)))
                    img = cv2.resize(img, (width // scale, height // scale))
                    row_cache.append(img)
                img_cache.append(np.concatenate(row_cache, axis=1))
                row_cache = []
            img = np.concatenate(img_cache, axis=0)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
            mask = 255 - th1
            cv2.imwrite(output_mask_path, mask)
            cv2.imwrite(output_img_path,img)
            cp('(#g)save_mask_path:{}(#)'.format(output_mask_path))
            #cp('(#g)save_orign_path:{}(#)'.format(output_img_path))

    with procedure('Cut image:\t{}'.format(img_path.split('/')[-2])):
        if not os.path.isdir(output_patch_path):
            mask = np.array(Image.open(output_mask_path))
            assert  len(mask.shape) == 2 ,print('size error')
            mask_patch_size = patch_size // scale
            step = mask_patch_size // 2
            try:
                os.makedirs(output_patch_path)
            except:
                pass
            grid = cont_overlap(mask,img_path.split('/')[-2], mask_patch_size, step, scale, patch_size, m, n)

            threshold = get_threshold(img_path.split('/')[-2], nums, bin, thresh)
            cp('(#r)Processinf:{}\tThreshold:{}'.format(img_path.split('/')[-2],threshold))

            for x,y,cur_scale in grid:
                if cur_scale > threshold:
                    data['roi'].append([x, y, cur_scale])
                    patch = np.array(slide.read_region((y, x), 0, (patch_size, patch_size)).convert('RGB'))
                    patch_name = "{}_{}.jpg".format(x, y)
                    patch_path = os.path.join(output_patch_path, patch_name)
                    cv2.imwrite(patch_path, patch)
                    #cp('(#y)save_path:\t{}(#)'.format(patch_path))


def get_threshold(img_name, nums, bin, threshold):
    overlap_path =os.path.join( cfg.patch_overlap, img_name + '.json')
    bin_nums = int(1/bin)
    count_range = np.arange(bin_nums)/bin_nums
    count_range = count_range[::-1]
    with open(overlap_path,'r') as f:
        data = np.array(json.load(f))
    for each_range in count_range:
        index = np.where(data[:,2] > each_range)
        #if (len(index[0]) >= nums and each_range <= threshold) or each_range <= 0.2 :
        if len(index[0]) >= nums  or (each_range <= threshold and len(index[0]) > 30):
            return each_range

def cont_overlap (mask, slide_name, mask_patch_size, step, scale, patch_size, m, n):
    patch_overlap_count = []
    h, w = mask.shape[0], mask.shape[1]
    for i in range(0, h, step):
        for j in range(0, w, step):
            si = min(i, h - mask_patch_size)
            sj = min(j, w - mask_patch_size)
            si = max(0, si)  # 有可能h比size还要小
            sj = max(0, sj)
            x = min(scale * si, m - patch_size)
            y = min(scale * sj, n - patch_size)
            sub_img = mask[si: si + mask_patch_size, sj: sj + mask_patch_size]
            cur_scale = (np.sum(sub_img) // 255) / (sub_img.shape[0] * sub_img.shape[1])
            patch_overlap_count.append([x,y,cur_scale])
            if sj != j:
                break
        if si != i:
            break
    if not os.path.isdir(cfg.patch_overlap):
        os.makedirs(cfg.patch_overlap)
    overlap_count_path = os.path.join(cfg.patch_overlap, slide_name + '.json')
    with open(overlap_count_path, 'w') as f:
        json.dump(patch_overlap_count, f)
    cp('(#g)save_overlap_count:\t{}(#)'.format(overlap_count_path))
    return  patch_overlap_count

if __name__ == '__main__':
    args = get_args()
    rename()
    df = pd.read_excel(cfg.label_path)
    labels = df.values
    m = {}
    for val in labels:
        m[val[0]] = val[1]
    params = []
    idx = 0
    for root, dirs, filenames in os.walk(cfg.data_append_path):
        for each_tif in filenames:
            if '.tif' in each_tif:
                name = each_tif.split('.')[0]
                flag = 'pos' if m[int(name)] == 'Positive' else 'neg'
                path = os.path.join(root, each_tif)  # ./EDCP/data_append/1/1.tif
                out_patch_path = os.path.join(args.output, flag, name)  # ./EDCP_PATCH/pos/1/
                idx += 1
                params.append([path, args.size, args.scale, out_patch_path, args.patch_size, args.nums, args.bin, args.threshold])

    # print(idx)
    cp('(#b)total_img:\t{}(#)'.format(idx))
    pool = threadpool.ThreadPool(args.poolsize)
    requests = threadpool.makeRequests(work, params)
    [pool.putRequest(req) for req in requests]
    pool.wait()








