import os
import sys
import time
# import visdom
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.controller import Controller
from models.shared_cnn import SharedCNN,ArcFace,separate_resnet_bn_paras,FocalLoss
from utils.utils import AverageMeter, Logger
from utils.cutout import Cutout

parser = argparse.ArgumentParser(description='ENAS')

parser.add_argument('--search_for', default='macro', choices=['macro'])
parser.add_argument('--data_path', default='E:\\lfw-align-128', type=str)
parser.add_argument('--output_filename', default='ENAS', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--log_every', type=int, default=50)
parser.add_argument('--eval_every_epochs', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cutout', type=int, default=0)
parser.add_argument('--fixed_arc', action='store_true', default=None)

parser.add_argument('--MULTI_GPU', type=int, default=False)
parser.add_argument('--device', type=int,default=0)

parser.add_argument('--child_num_layers', type=int, default=20)
parser.add_argument('--child_out_filters', type=int, default=32)
parser.add_argument('--child_grad_bound', type=float, default=5.0)
parser.add_argument('--child_l2_reg', type=float, default=0.00025)
parser.add_argument('--child_num_branches', type=int, default=4)
parser.add_argument('--child_keep_prob', type=float, default=0.9)
parser.add_argument('--child_lr_max', type=float, default=0.05)
parser.add_argument('--child_lr_min', type=float, default=0.0005)
parser.add_argument('--child_lr_T', type=float, default=10)

parser.add_argument('--controller_lstm_size', type=int, default=64)
parser.add_argument('--controller_lstm_num_layers', type=int, default=1)
parser.add_argument('--controller_entropy_weight', type=float, default=0.0001)
parser.add_argument('--controller_train_every', type=int, default=1)
parser.add_argument('--controller_num_aggregate', type=int, default=20)
parser.add_argument('--controller_train_steps', type=int, default=50)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_tanh_constant', type=float, default=1.5)
parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5)
parser.add_argument('--controller_skip_target', type=float, default=0.4)
parser.add_argument('--controller_skip_weight', type=float, default=0.8)
parser.add_argument('--controller_bl_dec', type=float, default=0.99)

args = parser.parse_args()

# vis = visdom.Visdom()
# vis.env = 'ENAS_' + args.output_filename
# vis_win = {'shared_cnn_acc': None, 'shared_cnn_loss': None, 'controller_reward': None,
#            'controller_acc': None, 'controller_loss': None}
PIXEL_MEANS =(0.485, 0.456, 0.406)  #RGB format mean and variances
PIXEL_STDS = (0.229, 0.224, 0.225)

validation_split = 0.1
shuffle_dataset = True
test_split = 0.1
random_seed = 42
class_num = 10

# LOSS = FocalLoss()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

controller = Controller(search_for=args.search_for,
                            search_whole_channels=True,
                            num_layers=args.child_num_layers,
                            num_branches=args.child_num_branches,
                            out_filters=args.child_out_filters,
                            lstm_size=args.controller_lstm_size,
                            lstm_num_layers=args.controller_lstm_num_layers,
                            tanh_constant=args.controller_tanh_constant,
                            temperature=None,
                            skip_target=args.controller_skip_target,
                            skip_weight=args.controller_skip_weight)
controller = controller.cuda()

shared_cnn = SharedCNN(num_layers=args.child_num_layers,
                           num_branches=args.child_num_branches,
                           out_filters=args.child_out_filters,
                           keep_prob=args.child_keep_prob)
shared_cnn = shared_cnn.cuda()

controller.eval()
controller()
sample_arc = controller.sample_arc

x = torch.tensor([50,3,112,112])
print(x.shape)
features = shared_cnn(x, sample_arc)