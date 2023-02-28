import torch
import numpy as np
import random
import os
import errno
from typing import Optional
from torch.optim.optimizer import Optimizer
import logging
import sys


def momentum_update(ema, current):
    lambd = np.random.uniform()
    return ema * lambd + current * (1 - lambd)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LrScheduler:
    def __init__(self, optimizer: Optimizer, max_iter, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter = max_iter

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
                print('wrong')
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1
