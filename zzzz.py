import torch
import torch.nn
import os
from torch.utils.tensorboard import SummaryWriter
from network.sst import *


def fun1(z):
    z.append(1)


def fun2(z):
    z.append(2)


z = list()
fun1(z)
fun2(z)

print(z)
