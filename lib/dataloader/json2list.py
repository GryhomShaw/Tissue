import os
import numpy as np
import json
from lib.config import config

with open(config.DATASET.SPLIT, 'r') as f:
    data = json.load(f)
print(len(data['train_pos']), len(data['val_pos']), len(data['test_pos']))
for each_phase in ['train', 'val', 'test']:
    cur = data[each_phase+'_'+'pos'] + data[each_phase+'_'+'neg']
    pos_list = []
    neg_list = []
    cur_name = each_phase + '.lst'
    for each_slide in data[each_phase+'_'+'pos']:
        for each_patch in os.listdir(each_slide):
            if '_mask' not in each_patch:
                pos_list.append(os.path.join(each_slide, each_patch))
    for each_slide in data[each_phase+'_' + 'neg']:
        for each_patch in os.listdir(each_slide):
            if '_mask' not in each_patch:
                neg_list.append(os.path.join(each_slide, each_patch))
    print("init:{} \t{}".format(len(pos_list), len(neg_list)))
    pos_len = len(pos_list)
    neg_sample_len = int(pos_len * 0.2)
    np.random.shuffle(neg_list)
    neg_list = neg_list if neg_sample_len >= len(neg_list) else neg_list[:neg_sample_len]
    img_list = pos_list + neg_list
    print(len(img_list), len(pos_list), len(neg_list))
    with open(cur_name, 'w') as f:
        for each_img in img_list:
            print("{}\t{}".format(each_img, each_img.replace('.jpg', '_mask.jpg')), file=f)

res = []
for each_phase in ['val', 'test']:
    cur = data[each_phase+'_pos']
    root = '/home/gryhomshaw/SSD1T/xiaoguohong/HRNet-Semantic-Segmentation/data/tissue/test'
    for each_slide in cur:
        slide_name = each_slide.split('/')[-1].replace('.jpg', '')
        res.append(os.path.join(root, slide_name))

with open('valtest.lst', 'w') as f:
    for each_slide in res:
        print(each_slide)
        print(each_slide, file=f)

