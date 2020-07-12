#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 08:57:15 2019
Train Mobilefacenet

@author: AIRocker
"""
import sys
sys.path.append('..')
import numpy as np
import torch
import torchvision.transforms as transforms
from data_set.dataloader import LFW, CFP_FP, AgeDB30
import time
from Evaluation import getFeature, evaluation_10_fold
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
def load_test_data(batch_size, dataset = 'Faces_emore'):
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # range [0.0, 1.0] -> [-1.0,1.0]

    root = 'E:\\dataset/LFW/lfw_align_112'
    file_list = 'E:\\dataset/LFW/pairs.txt'
    dataset_LFW = LFW(root, file_list, transform=transform)

    root = 'E:\\dataset/shunde'
    file_list = 'E:\\dataset/shunde/pairs.txt'
    dataset_privacy= CFP_FP(root, file_list, transform=transform)
    
    root = 'E:\\dataset/CFP-FP/CFP_FP_aligned_112'
    file_list = 'E:\\dataset/CFP-FP/cfp_fp_pair.txt'
    dataset_CFP_FP = CFP_FP(root, file_list, transform=transform)
        
    root = 'E:\\dataset/AgeDB-30/agedb30_align_112'
    file_list = 'E:\\dataset/AgeDB-30/agedb_30_pair.txt'
    dataset_AgeDB30 = AgeDB30(root, file_list, transform=transform)  
    
    dataloaders = {'LFW': DataLoaderX(dataset_LFW, batch_size=batch_size, shuffle=False, num_workers=2),
                   'privacy': DataLoaderX(dataset_privacy, batch_size=batch_size, shuffle=False, num_workers=2),
                   'CFP_FP': DataLoaderX(dataset_CFP_FP, batch_size=batch_size, shuffle=False, num_workers=2),
                   'AgeDB30': DataLoaderX(dataset_AgeDB30, batch_size=batch_size, shuffle=False, num_workers=2)}
    
    dataset = {'LFW': dataset_LFW,'CFP_FP': dataset_CFP_FP, 'AgeDB30': dataset_AgeDB30,'privacy':dataset_privacy}
    
    dataset_sizes = {'LFW': len(dataset_LFW),'privacy':len(dataset_privacy),
                     'CFP_FP': len(dataset_CFP_FP), 'AgeDB30': len(dataset_AgeDB30)}
    
    print('training and validation data loaded')
    
    return dataloaders, dataset_sizes, dataset

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from face_model import MobileFaceNet, l2_norm
    dataloaders , dataset_sizes, dataset = load_test_data(50)
    model = MobileFaceNet(512).to(device)
    state_dict = torch.load('arc_center_focal\Iter_555000_model.ckpt')['net_state_dict']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    # load params
    model.load_state_dict(state_dict)
    # print(state_dict)
    # model.drop_path_prob = 0
    model.eval()
    
    for phase in ['LFW', 'CFP_FP', 'AgeDB30','privacy']:                 
        featureLs, featureRs = getFeature(model, dataloaders[phase], device, flip = True)
        # ACCs, threshold = evaluation_10_fold(featureLs, featureRs, dataset[phase], method = 'l2_distance')
        ACCs, threshold = evaluation_10_fold(featureLs, featureRs, dataset[phase], method = 'cos_distance')
        print('Epoch {}/{}，{} average acc:{:.4f} average threshold:{:.4f}'.format(1, 1, phase, np.mean(ACCs) * 100, np.mean(threshold)))
                    
        
