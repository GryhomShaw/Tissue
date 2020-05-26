import os

from yacs.config import CfgNode as CN 

_C = CN()
_C.ROOT = '/home/gryhomshaw/SSD1T/xiaoguohong/MIL_Tissue'
_C.RANDOMSEED = 2333
_C.WORKERS = 16
_C.GPUS = (0, 1)
_C.PIN_MEMORY = True
_C.MODEL = 'densenet'
_C.NUMCLASSES = 2

#dataset

_C.DATASET = CN()
#gen_patch
_C.DATASET.NAME = 'Colon'
_C.DATASET.ROOT = os.path.join('/home/gryhomshaw/SSD1T/xiaoguohong/MIL_Tissue/data/', _C.DATASET.NAME)
_C.DATASET.POS = os.path.join(_C.DATASET.ROOT, 'pos')
_C.DATASET.NEG = os.path.join(_C.DATASET.ROOT, 'neg')

_C.DATASET.PATCH = os.path.join(_C.ROOT, 'patch_train', _C.DATASET.NAME)
_C.DATASET.ONLYPOS = False
_C.DATASET.USEMASK = False
_C.DATASET.SAVEMASK = False
_C.DATASET.SAVECOLOR = False
_C.DATASET.CHECK = True
_C.DATASET.PATCHSIZE = 54
_C.DATASET.POSSTEP = 27
_C.DATASET.NEGSTEP = 27
_C.DATASET.POSTHRESH = 0.5
_C.DATASET.NEGTHRESH = 0.3
_C.DATASET.POOLSIZE = 16
_C.DATASET.LOWWER = 5
_C.DATASET.UPPER = 200


_C.DATASET.MULTISCALE = [1, 2]
_C.DATASET.OVERLAP = 0.5
_C.DATASET.MEAN = [0.485, 0.456, 0.406] #RGB
_C.DATASET.STD = [0.229, 0.224, 0.225]  #RGB
_C.DATASET.ALPHA = [1.0, 1.0] #正负样本比
# tainval split
_C.DATASET.SPLIT = os.path.join(_C.ROOT, 'lib/dataloader/{}_train_val_split.json'.format(_C.DATASET.NAME))
_C.DATASET.SPLITRATIOS = [8, 2, 0]

_C.TRAIN = CN()
_C.TRAIN.VAL = True
_C.TRAIN.VALGAP = 1
_C.TRAIN.OUTPUT = os.path.join('./train_output', _C.DATASET.NAME)
_C.TRAIN.EPOCHS = 100
_C.TRAIN.BATCHSIZE = 128
_C.TRAIN.RESUME = False
_C.TRAIN.LR = 1e-5
_C.TRAIN.WD = 1e-5
_C.TRAIN.SELECTNUM = 2
_C.TRAIN.MODE = 'max-random'
_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''
_C.TRAIN.DISPLAY = 50
# loss
_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.NAME = 'focalloss'
_C.TRAIN.LOSS.WEIGHT = [0.5, 0.5]
_C.TRAIN.LOSS.GAMMA = 0

_C.TEST = CN()
_C.TEST.BATCHSIZE = 256
_C.TEST.OUTPUT = os.path.join('./test_output', _C.DATASET.NAME)
_C.TEST.CHECKPOINT = ''
_C.TEST.RESUME = True
_C.TEST.MULTISCALE = True
_C.TEST.DISPLAY = 50
_C.TEST.CAM = True
_C.TEST.CAMPATH = os.path.join(_C.TEST.OUTPUT, 'cam')

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