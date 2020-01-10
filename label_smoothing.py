import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import pdb 

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0, run_softmax=True):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.run_softmax = run_softmax
        
    def forward(self, x, target):
        if self.run_softmax:
            x = F.log_softmax(x, dim=-1)
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        if self.padding_idx!=-1:
            true_dist.fill_(self.smoothing / (self.size - 2)) # including padding token 
        else:
            true_dist.fill_(self.smoothing / (self.size - 1)) # no padding token 
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        if self.padding_idx!=-1:
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx)
            if (mask.sum()>0 and len(mask)>0): 
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
