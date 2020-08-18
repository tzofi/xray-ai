from __future__ import print_function

import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def xray_accuracy(output, target):
    auc_scores = []
    top_f1s = []
    top_accs = []
    for i in range(14):
        t = []
        p = []
        for sample in range(len(target)):
            if target[sample][i] != -1:
                t.append(target[sample][i])
                p.append(output[sample][i])
        t = np.array(t)
        #t[t > 0.5] = 1
        p = np.array(p)
        try:
            score = roc_auc_score(t, p)
            auc_scores.append(score)
        except Exception as e:
            print(t)
            auc_scores.append(np.nan)
        f1s = []
        accs = []
        for thresh_idx in range(0, 20, 1):
            thresh = thresh_idx * 0.05
            p_temp = np.copy(p)
            p_temp[p_temp >= thresh] = 1
            p_temp[p_temp < thresh] = 0
            f1s.append(metrics.f1_score(t, p_temp))
            accs.append(metrics.accuracy_score(t, p_temp))
        top_f1s.append(np.amax(f1s))
        top_accs.append(np.amax(accs))
        #best_threshs.append([0.05 * np.argmax(f1s), 0.05 * np.argmax(accs)])

    return auc_scores, top_f1s, top_accs
