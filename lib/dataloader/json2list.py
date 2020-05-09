import os
import numpy as np
import json
from lib.config import config

with open(config.DATASET.SPLIT, 'r') as f:
    data = json.load(f)

for each_phase in ['train', 'val', 'test']:
    cur = data[each_phase+'_'+'pos'] + data[each_phase+'_'+'neg']
    img_list = []
    cur_name = each_phase + '.lst'
    for each_slide in cur:
        for each_patch in os.listdir(each_slide):
            if '_mask' not in each_patch:
                img_list.append(os.path.join(each_slide, each_patch))
    print(len(img_list))
    with open(cur_name, 'w') as f:
        for each_img in img_list:
            print("{}\t{}".format(each_img, each_img.replace('.jpg', '_mask.jpg')), file=f)
