import os
import numpy as np
import cv2
from lib.config import config


def work(img_path, output_path, onlypos=False):
    slide_name = img_path.split('/')[-1]
    orign_path = os.path.join(config.DATASET.POS, slide_name+'.jpg')
    print(orign_path)
    orign_img = cv2.imread(orign_path)
    ostu_mask = ostu(orign_img)
    h, w = orign_img.shape[:2]
    class_list = ['pos'] if onlypos else ['pos', 'neg']
    scale_list = ['128', '256']
    mask = np.zeros([h, w])
    count = np.ones([h, w])
    for each_scale in scale_list:
        for each_class in class_list:
            cur_class_path = os.path.join(img_path, each_scale, each_class)
            for each_patch in os.listdir(cur_class_path):
                if '_mask' not in each_patch:
                    continue
                cur_h = int(each_patch.split('_')[0])
                cur_w = int(each_patch.split('_')[1].replace('_mask.jpg', ''))
                cur_patch = cv2.imread(os.path.join(cur_class_path, each_patch), 0)
                cur_patch_size = cur_patch.shape[0]
                mask[cur_h:cur_h + cur_patch_size, cur_w:cur_w+cur_patch_size] += cur_patch
                count[cur_h:cur_h + cur_patch_size, cur_w:cur_w+cur_patch_size] += 1
        mask = mask / count

        mask = np.where(ostu_mask == 0, mask*0, mask)
        mask = np.where(mask < 5, mask*0, mask + 255)
        mask = np.clip(mask, 0, 255)
        mask.astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, slide_name+'_mask.jpg'), mask)


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
    for each_img in os.listdir(root_path)[1:30]:
        work(os.path.join(root_path, each_img), output_path)
