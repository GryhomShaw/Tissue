import torch
import time
import numpy as np
import torch.nn.functional as F
from lib.utils.average import AverageMeter, ProgressMeter
from sklearn.metrics import f1_score
from eic_utils import cp
from lib.config import config


def trainer(run, loader, model, criterion, optimizer, writer):
    losses = AverageMeter('Loss', ':.4f')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(len(loader), [batch_time, data_time, losses],
                             prefix=cp.trans("(#b)[TRN](#) Epoch: [{}]".format(run)))
    model.train()
    running_loss = 0.
    end = time.time()
    criterion.adjust_gamma(gamma_ploy(run))
    for i, (input, target) in enumerate(loader):
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        losses.update(loss.item(), input.size(0))
        running_loss += loss.item()*input.size(0)
        progress.display(i)
        writer.add_scalar('train/loss', loss.item(), run * len(loader) + i)
        end =time.time()
    return running_loss/len(loader.dataset)


def inference(run, loader, model):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(len(loader), [batch_time, data_time],
                             prefix=cp.trans("(#b)[INF](#) Epoch: [{}]".format(run)))
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        end = time.time()
        for i, (input, _, _, _) in enumerate(loader):
            data_time.update(time.time() - end)
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            batch_time.update(time.time() - end)
            probs[i*config.TRAIN.BATCHSIZE:i*config.TRAIN.BATCHSIZE+input.size(0)] = output.detach()[:, 1].clone()
            if (i+1) % config.TRAIN.DISPLAY == 0:
                progress.display(i)
            end = time.time()
    return probs.cpu().numpy()


def inference_vt(epoch, loader, model):   #using in val and test
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(len(loader), [batch_time, data_time],
                             prefix=cp.trans("(#b)[INF](#) Epoch: [{}]").format(epoch))
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset), 2)
    img_idxs = []
    rows = []
    cols = []
    with torch.no_grad():
        end = time.time()
        for i, (input, img_idx, row, col) in enumerate(loader):
            data_time.update(time.time() - end)
            input = input.cuda()
            img_idxs.extend(list(img_idx.numpy()))
            rows.extend(list(row.numpy()))
            cols.extend(list(col.numpy()))
            output = F.softmax(model(input), dim=1)
            batch_time.update(time.time() - end)
            probs[i*config.TEST.BATCHSIZE:i*config.TEST.BATCHSIZE+input.size(0), :] = output.detach().clone()
            if (i+1) % config.TEST.DISPLAY == 0:
                progress.display(i)
            end = time.time()
    return probs.cpu().numpy(), np.array(img_idxs), np.array(rows), np.array(cols)


def gamma_ploy(epoch):
    if epoch < 2:
        return 0
    else:
        return 1