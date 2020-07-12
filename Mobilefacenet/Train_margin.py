#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 08:57:15 2019
Train Mobilefacenet

@author: AIRocker
"""
import os 
import sys
sys.path.append('..')
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler
from data_set.dataloader import LFW, CFP_FP, AgeDB30, CASIAWebFace, MS1M
from face_model import MobileFaceNet, Arcface
import time
from Evaluation import getFeature, evaluation_10_fold
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import torch.nn as nn
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Face_Detection_Training')
    parser.add_argument('--dataset', type=str, default='Faces_emore', help='Training dataset: CASIA, Faces_emore')
    parser.add_argument('--feature_dim', type=int, default=512, help='the feature dimension output')
    parser.add_argument('--batch_size', type=int, default=140, help='batch size for training and evaluation')
    parser.add_argument('--epoch', type=int, default=15, help='number of epoches for training')
    parser.add_argument('--method', type=str, default='l2_distance', 
                            help='methold to evaluate feature similarity, l2_distance, cos_distance')
    parser.add_argument('--flip', type=str, default=True, help='if flip the image with time augmentation')
    parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    from load_ms1m_lfw_agedb_cfp_dataset import *  
    dataloaders , dataset_sizes, dataset = load_glint_data_train_from_lmdb(args.batch_size, dataset = args.dataset)
    model = MobileFaceNet(args.feature_dim).to(device)  # embeding size is 512 (feature vector)
    
    from face_model import ArcMarginProduct
    margin = ArcMarginProduct(512,180855).to(device)
    model.load_state_dict(torch.load('/data/face_recognition/Mobilefacenet/arc_center_focal/Iter_555000_model.ckpt')['net_state_dict'])
    # margin.load_state_dict(torch.load('/data/face_recognition/Mobilefacenet/label_smooth_ckpt/Iter_290000_margin.ckpt')['net_state_dict'])

    
    criterion_focal = FocalLoss().to(device) #FocalLoss()
    criterion_center = CenterLoss(num_classes=180855, feat_dim=512, use_gpu=True).to(device)
    optimizer_centloss = torch.optim.SGD(criterion_center.parameters(), lr=0.5)
    
    # ！！！！！！！！！！！！！！只训练margin
    optimizer_ft = optim.SGD([{'params': margin.parameters(), 'weight_decay': 5e-4}], lr=0.001, momentum=0.9, nesterov=True)
    
    
    
    exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, float(args.epoch))
    start = time.time()
    ## save logging and weights
    train_logging_file = 'train_{}_logging.txt'.format('glint')
    test_logging_file = 'test_{}_logging.txt'.format('glint')
    save_dir = 'saving_{}_ckpt'.format(args.dataset)
    if os.path.exists(save_dir):
        pass
    else:
        os.makedirs(save_dir)

    best_acc = {'LFW': 0.0, 'CFP_FP': 0.0, 'AgeDB30': 0.0,'privacy':0.0}
    best_iters = {'LFW': 0, 'CFP_FP': 0, 'AgeDB30': 0,'privacy':0}
    total_iters = 0
    print('training kicked off..')
    print('-' * 10) 
    for epoch in range(args.epoch):
        # train model
        exp_lr_scheduler.step()
        margin.train()     
        since = time.time()

        for det in dataloaders['train_dataset']: 
            img, label = det[0].to(device), det[1].to(device)
            optimizer_ft.zero_grad()
            optimizer_centloss.zero_grad()
            with torch.set_grad_enabled(True):
                raw_logits = model(img)
                output = margin(raw_logits, label)
                loss_arc = criterion_focal(output, label)
                loss_center = criterion_center(raw_logits, label)
                loss_center *= 0.007
                loss = loss_arc + loss_center
                loss.backward()
                optimizer_ft.step()
                optimizer_centloss.step()
                # by doing so, weight_cent would not impact on the learning of centers
                for param in criterion_center.parameters():
                    param.grad.data *= (1. / 0.007)
                
                total_iters += 1
                # print train information
                if total_iters % 100 == 0:
                    # current training accuracy 
                    _, preds = torch.max(output.data, 1)
                    total = label.size(0)
                    correct = (np.array(preds.cpu()) == np.array(label.data.cpu())).sum()                  
                    time_cur = (time.time() - since) / 100
                    since = time.time()
                    
                    for p in  optimizer_ft.param_groups:
                        lr = p['lr']
                    print("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), correct/total, time_cur, lr))
                    with open(train_logging_file, 'a') as f:
                        f.write("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), correct/total, time_cur, lr)+'\n')
                    f.close()
                    
            # save model
            if total_iters % 100 == 0:
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': margin.state_dict()},
                    os.path.join(save_dir, 'Iter_glint_margin.ckpt'))
            
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        