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
class NetworkImageNet(nn.Module):
    
  def __init__(self, C, num_classes, layers,genotype):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.PReLU(C // 2),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.PReLU(C),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      


    self.linear7 = ConvBlock(C_prev, C_prev, (7, 7), 1, 0, dw=True, linear=True)
    self.dropout = nn.Dropout(0.5)
    self.conv_6_flatten = Flatten()
    self.linear = nn.Linear(C_prev, 512, bias=False)
    self.bn = nn.BatchNorm1d(512)

  def forward(self, input):
    
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      
    
    print(s1.shape)
    import sys 
    sys.exit()
    out = self.linear7(s1)
    out = self.dropout(out)
    out = self.conv_6_flatten(out)
    out = self.linear(out)
    out = self.bn(out)
    return out

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # from face_model import MobileFaceNet, l2_norm
    from collections import namedtuple
    dataloaders , dataset_sizes, dataset = load_test_data(50)
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    genotype = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
    from face_search.model_search_ms1m import Network
    model =NetworkImageNet(48, 85742, 14, genotype).to(device)
    # print(model)/data/face_recognition/1MobileFaceNet_Tutorial_Pytorch-master1/saving_Faces_emore_ckpt/Iter_009000_margin.ckpt
    checkpoint=torch.load('best_arc64\Iter_725000_model.ckpt')
    # model = MobileFaceNet(512).to(device)
    # state_dict = torch.load('arc_center_focal\Iter_555000_model.ckpt')['net_state_dict']
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
                    
        
