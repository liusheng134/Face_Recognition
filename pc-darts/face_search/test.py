import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import copy

from torch.autograd import Variable
from model_search_ms1m import Network

criterion = nn.CrossEntropyLoss()

model = Network(32, 10, 8, criterion)

x = torch.zeros(2,3,112,112)

out = model(x)

print(out.shape)