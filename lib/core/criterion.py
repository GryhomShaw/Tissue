import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import f1_score


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def adjust_gamma(self, gamma):
        self.gamma = gamma

    def forward(self, input, target):

        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        self.logpt = F.log_softmax(input, dim=1)
        self.logpt = self.logpt.gather(1, target)
        self.logpt = self.logpt.view(-1)
        self.pt = self.logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            self.logpt = self.logpt * at

        self.loss = -1 * (1-self.pt)**self.gamma * self.logpt
        if self.size_average:
            return self.loss.mean()
        else:
            return self.loss.sum()


def calc_err(pred, real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred == 1, neq).sum()) / (real == 0).sum()
    fnr = float(np.logical_and(pred == 0, neq).sum()) / (real == 1).sum()
    f1 = f1_score(real, pred, average='binary')
    return err, fpr, fnr, f1


def calc_dsc(pred, mask):
    assert pred.shape == mask.shape, print("SHAPE ERROR")
    assert np.max(pred) <= 1 and np.max(mask) <= 1, print('VAL ERROR')
    eq = np.equal(pred, mask)
    overlap = 2 * (np.logical_and(eq, pred == 1).sum())
    return overlap / (pred.sum() + mask.sum())
