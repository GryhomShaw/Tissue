import numpy as np
import os
import json
from lib.config import config

np.random.seed(config.RANDOMSEED)
data_pos = os.path.join(config.DATASET.PATCH, 'pos')
data_neg = os.path.join(config.DATASET.PATCH, 'neg')
save_path = config.DATASET.SPLIT
def train_val_split(path):
    # list all image except mask
    images = [os.path.join(path, each) for each in os.listdir(path)]    # ./ECDP_PATCH/pos/1
    ratios = config.DATASET.SPLITRATIOS
    # random shuffle list
    np.random.shuffle(images)

    ratios = np.array(ratios)
    percent = ratios / ratios.sum()

    total = len(images)

    train_n = int(round(total * percent[0]))

    return images[:train_n], images[train_n:]
    
train_neg_list, val_neg_list = train_val_split(data_neg)
print(list(map(len, [train_neg_list, val_neg_list])))
train_pos_list, val_pos_list = train_val_split(data_pos)
print(list(map(len, [train_pos_list, val_pos_list])))

with open(save_path, 'w') as f:
    json.dump({
        'train_pos': train_pos_list,
        'train_neg': train_neg_list,
        'val_pos' : val_pos_list,
        'val_neg': val_neg_list,
    }, f)


