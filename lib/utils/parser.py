import os
import cv2
import numpy as np
from lib.config import config


def probs_parser(probs, img_idxs, rows, cols, dset, scale):
    assert isinstance(probs, np.ndarray) and isinstance(img_idxs, np.ndarray) and isinstance(rows, np.ndarray) and \
           isinstance(cols, np.ndarray), print("TYPE ERROR")

    assert probs.shape[0] == img_idxs.shape[0] and img_idxs.shape[0] == rows.shape[0] and \
           rows.shape[0] == cols.shape[0], print("LENGTH ERROR")

    prefix = 'tissue-train-'
    slide_len = np.array(dset.slideLen[:]) * pow(scale, 2)
    assert slide_len[-1] == probs.shape[0], print("VAL ERROR")
    slide_names = [prefix + dset.grid[each_idx].split('/')[-3] + '/' + dset.grid[each_idx].split('/')[-2]+'.jpg'
                    for each_idx in img_idxs]
    row_offsets = np.array([int(dset.grid[each_idx].split('/')[-1].split('_')[0]) for each_idx in img_idxs])
    col_offsets = np.array([int((dset.grid[each_idx].split('/')[-1].split('_')[-1]).replace('.jpg', ''))
                            for each_idx in img_idxs])
    img_path = [os.path.join(os.path.join(config.DATASET.ROOT, each_name)) for each_name in slide_names]
    rows = rows + row_offsets
    cols = cols + col_offsets
    res = {}
    for idx in range(slide_len.shape[0]-1):
        start = slide_len[idx]
        end = slide_len[idx+1]
        res[img_path[start]] = []
        for label_idx in range(start, end):
            res[img_path[start]].append([rows[label_idx], cols[label_idx], probs[label_idx], scale])
    return res


def group_max(slideLen, data, nmax, scale):
    groups = []
    slideLen = np.array(slideLen[:]) * pow(scale, 2)
    for slide_idx in np.arange(1, len(slideLen)):
        groups.extend([slide_idx-1] * (slideLen[slide_idx] - slideLen[slide_idx-1]))
    groups = np.array(groups)
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out


def group_argtopk(data, targets, slideLen, scale):
    k = config.TRAIN.SELECTNUM * scale
    slideLen = np.array(slideLen[:]) * pow(scale, 2)
    groups = []
    for slide_idx in np.arange(1, len(slideLen)):
        groups.extend([slide_idx-1] * (slideLen[slide_idx] - slideLen[slide_idx-1]))
    groups = np.array(groups)
    assert groups.shape[0] == slideLen[-1], print("SHAPE ERROR")
    assert groups.shape[0] == data.shape[0], print("SHAPE ERROR")
    order = np.lexsort((data, groups))
    groups = groups[order]
    index = np.full(len(groups), False)
    if config.TRAIN.MODE == 'max-max': # max-max
        index[-k:] = True
        index[:-k] = groups[k:] != groups[:-k]
    else:
        for idx in range(1, slideLen.shape[0]):
            cur_id = idx-1
            if targets[cur_id] == 1:
                index[slideLen[idx] - k:slideLen[idx]] = True
            else:
                index[slideLen[cur_id]:slideLen[cur_id] + k] = True
    return list(order[index])


def get_mask(patch_info):
    res = {}
    for each_img, labels in patch_info.items():
        img = cv2.imread(each_img)
        h, w = img.shape[0], img.shape[1]
        mask = np.zeros(shape=[h, w, 2]).astype(np.float)
        count = np.ones(shape=[h, w, 1])
        for each_label in labels:
            patch_len = config.DATASET.PATCHSIZE // each_label[-1]
            x1 = each_label[0]
            y1 = each_label[1]
            x2 = x1 + patch_len
            y2 = y1 + patch_len
            mask[x1:x2, y1:y2, :] += each_label[2]
            count[x1:x2, y1:y2, :] += 1
        mask = mask / count
        mask = np.argmax(mask, axis=2)
        res[each_img] = mask
    return res
