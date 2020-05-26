import os
import numpy as np
import cv2
from lib.config import config
from dcrf import crf_inference


def work(img_path, output_path, use_dcrf=False):
    slide_name = img_path.split('/')[-1]
    orign_path = os.path.join(config.DATASET.POS, slide_name+'.jpg')
    mil_mask_path = os.path.join('/home/gryhomshaw/SSD1T/xiaoguohong/MIL_Tissue/test_output/densenet/max-random_20200526_155430/pos',
                                  slide_name + '_mask.jpg')
    print(orign_path)
    orign_img = cv2.imread(orign_path)
    mil_mask = cv2.imread(mil_mask_path, 0)
    mil_mask = (mil_mask // 255).astype(np.uint8)
    ostu_mask = ostu(orign_img)
    h, w = orign_img.shape[:2]

    scale_list = ['128', '256']
    mask = np.zeros([h, w, 2])
    count = np.ones([h, w, 2])
    for each_scale in scale_list:
        cur_path = os.path.join(img_path, each_scale)
        for each_patch in os.listdir(cur_path):
            if '.npy' not in each_patch:
                continue
            cur_h = int(each_patch.split('_')[0])
            cur_w = int(each_patch.split('_')[1].replace('.npy', ''))
            cur_patch = np.load(os.path.join(cur_path, each_patch))
            # cur_patch = cv2.imread(os.path.join(cur_path, each_patch), 0)
            cur_patch_size = cur_patch.shape[0]
           # print(cur_patch.shape)
            mask[cur_h:cur_h + cur_patch_size, cur_w:cur_w+cur_patch_size] += cur_patch
            count[cur_h:cur_h + cur_patch_size, cur_w:cur_w+cur_patch_size] += 1

    mask = mask / count
    if use_dcrf:
        mask_crf = mask.copy()
        mask_crf = mask_crf.transpose(2, 0, 1)
        mask_crf = crf_inference(orign_img, mask_crf)
        mask_crf = mask_crf.transpose(1, 2, 0)
        pos_mask = mask_crf[:, :, 1]
    else:
        pos_mask = mask[:, :, 1]
    #print(np.max(pos_mask), np.max(mil_mask))
    pos_mask = np.where(ostu_mask == 0, pos_mask * 0, pos_mask)
    pos_mask = np.where(mil_mask == 0, pos_mask * 0, pos_mask)

    pos_mask = np.clip(pos_mask*255, 0, 255)
    pos_mask.astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, slide_name+'_mask.jpg'), pos_mask)


def ostu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    mask = 255 - th1
    return mask


if __name__ == '__main__':
    output_path = '/home/gryhomshaw/SSD1T/xiaoguohong/MIL_Tissue/test_output/cam_mask/'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    root_path = '/home/gryhomshaw/SSD1T/xiaoguohong/MIL_Tissue/test_output/cam'
    for each_img in os.listdir(root_path)[1:3]:
        work(os.path.join(root_path, each_img), output_path, True)
