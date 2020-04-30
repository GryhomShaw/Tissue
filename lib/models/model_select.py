import torchvision.models as models

MODEL_DICT = {
    'resnet34': models.resnet34(True),
    'resnet50': models.resnet50(True),
    'resnext': models.resnext50_32x4d(True)

}


def get_model(cfg):

    return MODEL_DICT[cfg.TRAIN.MODEL]