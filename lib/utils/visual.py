import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from lib.config import config


def vis_hist(data, bins=10, output_path="hist.png"):
    assert isinstance(data, np.ndarray), print('type error')
    plt.hist(data, bins, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel('nums')
    plt.ylabel('cnt')
    plt.savefig(output_path)


if __name__ == '__main__':
    data_path = config.DATASET.PATCH


    pos_list = []
    neg_list = []
    for each in ['pos', 'neg']:
        cur_path = os.path.join(data_path, each)
        for each_img in os.listdir(cur_path):
            if each == 'pos':
                pos_list.append(len(os.listdir(os.path.join(cur_path, each_img))))
            else:
                neg_list.append(len(os.listdir(os.path.join(cur_path, each_img))))

    pos_list = np.array(pos_list)
    neg_list = np.array(neg_list)
    #vis_hist(np.array(pos_list), output_path="../../hist_pos.png")
    #vis_hist(np.array(neg_list), output_path="../../hist_neg.png")
    counts, bin_edges = np.histogram(neg_list, bins=100)
    print(counts)
    print(bin_edges.astype(np.int))
    print(np.max(neg_list), np.min(neg_list))