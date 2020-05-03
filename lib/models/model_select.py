import torchvision.models as models
import torch.nn as nn
MODEL_DICT = {
    'resnet34': models.resnet34(True),
    'resnet50': models.resnet50(True),
    'resnext': models.resnext50_32x4d(True),
    'resnext101': models.resnext101_32x8d(True),
    'deneset' : models.densenet121(True)
}


def get_model(cfg):
    model = MODEL_DICT[cfg.MODEL]
    if 'deneset' in cfg.MODEL:
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    else:
        model.fc = nn.Linear(model.fc.in_features, 2)
    return model