import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor


class Record:
    def __init__(self):
        self.preds = []
        self.tgts = []

    def add(self, pred, tgt):
        """pred and tgt must be homogeneous tensors"""
        if pred.shape[-1] == 1: pred = pred.squeeze(-1) # [B]
        self.preds.append(pred)
        self.tgts.append(tgt)

    def clear(self):
        self.preds = []
        self.tgts = []

    def get_all(self):
        preds = torch.cat(self.preds, dim=0)
        tgts = torch.cat(self.tgts, dim=0)
        assert preds.shape == tgts.shape
        return preds, tgts

def roc_auc_score_rev(pred, tgt):
    """Flipping the signatures."""
    pred = pred.cpu().numpy()
    tgt = tgt.cpu().numpy()
    return roc_auc_score(tgt, pred)

def accuracy(pred: Tensor, tgt: Tensor) -> float:
    pred_label = (pred >= 0.5).long()
    correct = (tgt == pred_label).sum()
    total = len(tgt)
    acc = correct/total
    return acc.item()


