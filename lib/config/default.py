import os

from yacs.config import CfgNode as CN 


_C = CN()
_C.ROOT = '/home/gryhomshaw/SSD1T/xiaoguohong/MIL_Tissue'
_C.RANDOMSEED = 2333
_C.WORKERS = 16
_C.GPUS = "0,1"
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 20


#dataset
_C.DATASET = CN()
#gen_patch
_C.DATASET.ROOT = '/home/gryhomshaw/SSD1T/xiaoguohong/MIL_Tissue/data'
_C.DATASET.POS = os.path.join(_C.DATASET.ROOT, 'tissue-train-pos')
_C.DATASET.NEG = os.path.join(_C.DATASET.ROOT, 'tissue-train-neg')
_C.DATASET.PATCH = os.path.join(_C.ROOT, 'patch')
_C.DATASET.PATCHSIZE = 512
_C.DATASET.PATCHSTEP = _C.DATASET.PATCHSIZE
_C.DATASET.PATCHTHRESH = 0.65
_C.DATASET.POOLSIZE = 1
_C.DATASET.LOWWER = 20
_C.DATASET.UPPER = 100
_C.DATASET.MULTISCALE = [1, 2, 4]
_C.DATASET.MEAN = [0.485, 0.456, 0.406] #RGB
_C.DATASET.STD = [0.229, 0.224, 0.225]  #RGB
# tainval split
_C.DATASET.SPLIT = os.path.join(_C.ROOT, 'lib/dataloader/train_val_split.json')
_C.DATASET.SPLITRATIOS = [9, 1]

_C.TRAIN = CN()
_C.TRAIN.MODEL = 'resnext'
_C.TRAIN.VAL = True
_C.TRAIN.VALGAP = 3
_C.TRAIN.OUTPUT = './train_output'
_C.TRAIN.EPOCHS = 20
_C.TRAIN.BATCHSIZE = 32
_C.TRAIN.RESUME = False
_C.TRAIN.LR = 1e-5
_C.TRAIN.WD = 1e-5
_C.TRAIN.SELECTNUM = 2
_C.TRAIN.MODE = 'max-max'
_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''
# loss
_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.NAME = 'focalloss'
_C.TRAIN.LOSS.WEIGHT = [0.5, 0.5]
_C.TRAIN.LOSS.GAMMA = 0


def update_config(cfg, args):
    cfg.defrost()
    if args.cfg is not None:
        cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)