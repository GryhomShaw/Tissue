import os

from yacs.config import CfgNode as CN 


_C = CN()
_C.ROOT = '/home/gryhomshaw/SSD1T/xiaoguohong/MIL_Tissue'

#dataset

_C.DATASET = CN()
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