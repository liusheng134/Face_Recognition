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
from face_model import MobileFaceNet
import time
from Evaluation import getFeature, evaluation_10_fold
from enas.train import *
from enas.models.shared_cnn import SharedCNN,FixSharedCNN
from enas.models.controller import Controller
from enas.utils.cutout import Cutout
import torch.nn as nn
import math
import torch.nn.functional as F
import sys
class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """
    def __init__(self, in_features, out_features, device_id, s = 64.0, m = 0.50, easy_margin = False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
      
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, 1, dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1) 
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

def load_data(batch_size, dataset = 'Faces_emore'):
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((120, 120), interpolation=3),
        transforms.RandomCrop(112),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    train_transform.transforms.append(Cutout(length=20))

    root = '/data/face_recognition/LFW/lfw_align_112'
    file_list = '/data/face_recognition/LFW/pairs.txt'
    
    dataset_LFW = LFW(root, file_list, transform=transform)
    
    root = '/data/face_recognition/CFP-FP/CFP_FP_aligned_112'
    file_list = '/data/face_recognition/CFP-FP/cfp_fp_pair.txt'
    dataset_CFP_FP = CFP_FP(root, file_list, transform=transform)
        
    root = '/data/face_recognition/AgeDB-30/agedb30_align_112'
    file_list = '/data/face_recognition/AgeDB-30/agedb_30_pair.txt'
    dataset_AgeDB30 = AgeDB30(root, file_list, transform=transform)  
    
    if dataset == 'CASIA':
        
        root = 'data_set/CASIA_Webface_Image'
        file_list = 'data_set/CASIA_Webface_Image/webface_align_112.txt'
        dataset_train = CASIAWebFace(root, file_list, transform=transform)
        
    elif dataset == 'Faces_emore':

        root = '/data/zwh/00.PaperRecurrence/2.InsightFace_Pytorch/2.Dataset/faces_emore/imgs'
        file_list = '/data/zwh/00.PaperRecurrence/2.InsightFace_Pytorch/2.Dataset/faces_emore/imgs/faces_emore_align_112.txt'
        dataset_train = MS1M(root, file_list, transform=transform) 
    
    else:
        raise NameError('no training data exist!')

    dataset_size = len(dataset_train)
    train_size = int(0.8 * dataset_size)
    valid_size = dataset_size - train_size
    train_subset, valid_subset = torch.utils.data.random_split(dataset_train, [train_size, valid_size])
    
    dataloaders = {'train_dataset': data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4),
                   'train_subset': data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4),
                   'valid_subset': data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4),
                   'LFW': data.DataLoader(dataset_LFW, batch_size=batch_size, shuffle=False, num_workers=4),
                   'CFP_FP': data.DataLoader(dataset_CFP_FP, batch_size=batch_size, shuffle=False, num_workers=4),
                   'AgeDB30': data.DataLoader(dataset_AgeDB30, batch_size=batch_size, shuffle=False, num_workers=4)}
    
    dataset = {'train_dataset': dataset_train,'train_subset': train_subset,'valid_subset': valid_subset,'LFW': dataset_LFW,
               'CFP_FP': dataset_CFP_FP, 'AgeDB30': dataset_AgeDB30}
    
    dataset_sizes = {'train': len(dataset_train),'train_subset':len(train_subset),'valid_subset':len(valid_subset),'LFW': len(dataset_LFW),
                     'CFP_FP': len(dataset_CFP_FP), 'AgeDB30': len(dataset_AgeDB30)}
    
    print('training and validation data loaded')
    
    return dataloaders, dataset_sizes, dataset

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Face_Detection_Training')
    parser.add_argument('--dataset', type=str, default='Faces_emore', help='Training dataset: CASIA, Faces_emore')
    parser.add_argument('--feature_dim', type=int, default=512, help='the feature dimension output')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for training and evaluation')
    parser.add_argument('--epoch', type=int, default=12, help='number of epoches for training')
    parser.add_argument('--method', type=str, default='l2_distance', 
                            help='methold to evaluate feature similarity, l2_distance, cos_distance')
    parser.add_argument('--flip', type=str, default=True, help='if flip the image with time augmentation')
    parser.add_argument('--resume', default='./ENAS.pth.tar', type=str)

    args = parser.parse_args()
    
    sys.stdout = Logger(filename='logging/eans_fixed.log')
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  
    dataloaders , dataset_sizes, dataset = load_data(args.batch_size, dataset = args.dataset)
    GPU_ID = [1]
    # model = MobileFaceNet(args.feature_dim).to(device)  # embeding size is 512 (feature vector)
    ## save logging and weights
    train_logging_file = 'train_{}_logging.txt'.format(args.dataset)
    test_logging_file = 'test_{}_logging.txt'.format(args.dataset)
    save_dir = 'saving_{}_ckpt'.format(args.dataset)
    #################################################
    controller = Controller(search_for='macro',
                            search_whole_channels=True,
                            num_layers=20,
                            num_branches=4,
                            out_filters=32,
                            lstm_size=64,
                            lstm_num_layers=1,
                            tanh_constant=1.5,
                            temperature=None,
                            skip_target=0.4,
                            skip_weight=0.8)
    controller = controller#.to(device)

    

    shared_cnn = SharedCNN(num_layers=20,
                           num_branches=4,
                           out_filters=32,
                           keep_prob=0.9)
    shared_cnn = shared_cnn#.to(device)
    


    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L218
    controller_optimizer = torch.optim.Adam(params=controller.parameters(),
                                            lr=0.001,
                                            betas=(0.0, 0.999),
                                            eps=1e-3)

    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L213
    shared_cnn_optimizer = torch.optim.SGD(params=shared_cnn.parameters(),
                                           lr=0.05,
                                           momentum=0.9,
                                           nesterov=True,
                                           weight_decay=0.00025)
    
    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L154
    shared_cnn_scheduler = CosineAnnealingLR(optimizer=shared_cnn_optimizer,
                                             T_max=10,
                                             eta_min=0.0005)
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,map_location='cpu')
            start_epoch = checkpoint['epoch']
            # args = checkpoint['args']
            shared_cnn.load_state_dict(checkpoint['shared_cnn_state_dict'])
            controller.load_state_dict(checkpoint['controller_state_dict'])
            shared_cnn_optimizer.load_state_dict(checkpoint['shared_cnn_optimizer'])
            controller_optimizer.load_state_dict(checkpoint['controller_optimizer'])
            shared_cnn_scheduler.optimizer = shared_cnn_optimizer  # Not sure if this actually works
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError("No checkpoint found at '{}'".format(args.resume))
    else:
        start_epoch = 0
    best_arc, best_val_acc = get_best_arc(controller, shared_cnn, dataloaders, n_samples=100, verbose=False)
    print(best_arc)
    
    print('Best architecture:')
    print_arc(best_arc)
    print('Validation accuracy: ' + str(best_val_acc))

    fixed_cnn = FixSharedCNN(num_layers=20,
                          num_branches=4,
                          out_filters=512 // 32,  # args.child_out_filters
                          keep_prob=0.9,
                          fixed_arc=best_arc)
    model = fixed_cnn.to(device)
    margin = ArcFace(in_features = args.feature_dim, out_features = int(dataset['train_dataset'].class_nums), device_id = GPU_ID)

    fixed_cnn_optimizer = torch.optim.SGD([
        {'params': model.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}],
                                          lr=0.01,
                                          momentum=0.9,
                                          nesterov=True)

    fixed_cnn_scheduler = lr_scheduler.MultiStepLR(fixed_cnn_optimizer, milestones=[6, 8, 10], gamma=0.3) 
    #################################################

    print('MobileFaceNet face detection model loaded')
    
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # optimizer_ft = optim.SGD([
    #     {'params': model.parameters(), 'weight_decay': 5e-4},
    #     {'params': margin.parameters(), 'weight_decay': 5e-4}], lr=0.01, momentum=0.9, nesterov=True)

    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[6, 8, 10], gamma=0.3) 
    start = time.time()
    
    if not os.path.exists(save_dir):
        # raise NameError('model dir exists!')
        os.makedirs(save_dir)

    best_acc = {'LFW': 0.0, 'CFP_FP': 0.0, 'AgeDB30': 0.0}
    best_iters = {'LFW': 0, 'CFP_FP': 0, 'AgeDB30': 0}
    total_iters = 0
    print('training kicked off..')
    print('-' * 10) 
    for epoch in range(args.epoch):
        # train model
        fixed_cnn_scheduler.step()
        model.train()     
        since = time.time()
        for det in dataloaders['train_dataset']: 
            img, label = det[0].to(device), det[1].to(device)
            fixed_cnn_optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                raw_logits = model(img,best_arc)

                output = margin(raw_logits, label)
                loss = criterion(output, label)
                
                loss.backward()
                fixed_cnn_optimizer.step()
                
                total_iters += 1
                # print train information
                if total_iters % 100 == 0:
                    # current training accuracy 
                    _, preds = torch.max(output.data, 1)
                    total = label.size(0)
                    correct = (np.array(preds.cpu()) == np.array(label.data.cpu())).sum()                  
                    time_cur = (time.time() - since) / 100
                    since = time.time()
                    
                    for p in  fixed_cnn_optimizer.param_groups:
                        lr = p['lr']
                    print("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), correct/total, time_cur, lr))
                    with open(train_logging_file, 'a') as f:
                        f.write("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), correct/total, time_cur, lr)+'\n')
                    f.close()
                    
            # save model
            if total_iters % 3000 == 0:

                torch.save({
                    'iters': total_iters,
                    'net_state_dict': model.state_dict()},
                    os.path.join(save_dir, 'Iter_%06d_model.ckpt' % total_iters))
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': margin.state_dict()},
                    os.path.join(save_dir, 'Iter_%06d_margin.ckpt' % total_iters))
            
            # evaluate accuracy
            if total_iters % 3000 == 0:
                
                model.eval()
                for phase in ['LFW', 'CFP_FP', 'AgeDB30']:                 
                    featureLs, featureRs = getFeature(model,best_arc, dataloaders[phase], "cuda:1", flip = args.flip,)
                    ACCs, threshold = evaluation_10_fold(featureLs, featureRs, dataset[phase], method = args.method)
                    print('Epoch {}/{},{} average acc:{:.4f} average threshold:{:.4f}'
                          .format(epoch, args.epoch-1, phase, np.mean(ACCs) * 100, np.mean(threshold)))
                    if best_acc[phase] <= np.mean(ACCs) * 100:
                        best_acc[phase] = np.mean(ACCs) * 100
                        best_iters[phase] = total_iters
                    
                    with open(test_logging_file, 'a') as f:
                        f.write('Epoch {}/{}, {} average acc:{:.4f} average threshold:{:.4f}'
                                .format(epoch, args.epoch-1, phase, np.mean(ACCs) * 100, np.mean(threshold))+'\n')
                    f.close()
                    
                model.train()
                        
    time_elapsed = time.time() - start  
    print('Finally Best Accuracy: LFW: {:.4f} in iters: {}, CFP_FP: {:.4f} in iters: {} and AgeDB-30: {:.4f} in iters: {}'.format(
        best_acc['LFW'], best_iters['LFW'], best_acc['CFP_FP'], best_iters['CFP_FP'], best_acc['AgeDB30'], best_iters['AgeDB30']))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        
