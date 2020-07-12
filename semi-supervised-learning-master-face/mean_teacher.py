#coding=utf-8
import torch
from torch import  autograd,nn
from torch.utils.data import DataLoader, Dataset
from data_layer import mydata,make_weights_for_balanced_classes
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as function
import os
import time
from face_model import *
from Evaluation import *
import argparse

def update_ema_variables(model, ema_model,alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = function.softmax(input_logits, dim=1)
    target_softmax = function.softmax(target_logits, dim=1)
    num_classes = 512
    return function.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes
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
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))
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
        if self.use_gpu: classes = classes.to(device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

parser = argparse.ArgumentParser(description='Face_Detection_Training')
parser.add_argument('--dataset', type=str, default='Faces_emore', help='Training dataset: CASIA, Faces_emore')
parser.add_argument('--feature_dim', type=int, default=512, help='the feature dimension output')
parser.add_argument('--batch_size', type=int, default=200, help='batch size for training and evaluation')
parser.add_argument('--epoch', type=int, default=20, help='number of epoches for training')
parser.add_argument('--method', type=str, default='l2_distance', 
                            help='methold to evaluate feature similarity, l2_distance, cos_distance')
parser.add_argument('--flip', type=str, default=True, help='if flip the image with time augmentation')
args = parser.parse_args()

torch.backends.cudnn.enabled = False
torch.manual_seed(7)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
net_student=MobileFaceNet(args.feature_dim).to(device)
net_teacher=MobileFaceNet(args.feature_dim).to(device)
margin = ArcMarginProduct(args.feature_dim,85742).to(device)

# 加载模型参数
# net_student.load_state_dict(torch.load('./model/Iter_455000_model.ckpt')['net_state_dict'])
# net_teacher.load_state_dict(torch.load('./model/Iter_455000_model.ckpt')['net_state_dict'])
# margin.load_state_dict(torch.load('./model/Iter_455000_margin.ckpt')['net_state_dict'])



for param in net_teacher.parameters():
    param.detach_()

from dataloader import *
dataloaders , dataset_sizes, dataset = load_data_train_fix_from_lmdb(args.batch_size, dataset = args.dataset)
min_batch_size=args.batch_size
# weights = make_weights_for_balanced_classes(train_data.labels, 3)
# weights = torch.DoubleTensor(weights)
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
# train_dataloader=DataLoader(train_data,batch_size=min_batch_size,shuffle=True,num_workers=8)
# print train_dataloader

# valid_data=mydata('data/white/val.txt','./',is_train=False)
# valid_dataloader=DataLoader(valid_data,batch_size=min_batch_size,shuffle=True,num_workers=8)

# classify_loss_function = torch.nn.CrossEntropyLoss(size_average=False,ignore_index=-1).cuda()
criterion_focal = FocalLoss().to(device)
criterion_center = CenterLoss(num_classes=85742, feat_dim=args.feature_dim, use_gpu=True).to(device)

# optimizer = torch.optim.SGD(net_student.parameters(),lr = 0.001, momentum=0.9)
optimizer_centloss = torch.optim.SGD(criterion_center.parameters(), lr=0.5)
optimizer = torch.optim.SGD([
    {'params': net_student.parameters(), 'weight_decay': 5e-4},
    {'params': margin.parameters(), 'weight_decay': 5e-4}], lr=0.01, momentum=0.9, nesterov=True)

exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epoch))

train_logging_file = 'train_{}_logging_from_scrach.txt'.format(args.dataset)
test_logging_file = 'test_{}_logging_from_scrach.txt'.format(args.dataset)
save_dir = 'model_from_scrach'

if os.path.exists(save_dir):
    pass
else:
    os.makedirs(save_dir)

best_acc = {'LFW': 0.0, 'CFP_FP': 0.0, 'AgeDB30': 0.0,'privacy':0.0}
best_iters = {'LFW': 0, 'CFP_FP': 0, 'AgeDB30': 0,'privacy':0}
total_iters = 0

globals_step=0
for epoch in range(args.epoch):
    # train model
    exp_lr_scheduler.step()
    globals_classify_loss=0
    globals_consistency_loss = 0
    net_student.train()
    margin.train()
    start=time.time()
    end=0

    since = time.time()

    for det in dataloaders['train_dataset']: 
        net_student.train()
        margin.train
        img, label = det[0][0].to(device), det[1].to(device)

        optimizer.zero_grad()  #
        optimizer_centloss.zero_grad()  #

        # x_student=autograd.Variable(img[0]).to(device)
        # y=autograd.Variable(y).to(device)
        raw_logits_student=net_student(img)
        output = margin(raw_logits_student, label)

        loss_arc=criterion_focal(output, label)
        loss_center = criterion_center(raw_logits_student, label)
        loss_center *= 0.03
        

        with torch.no_grad():
            x_teacher= autograd.Variable(det[0][1]).to(device)
            predict_teacher = net_teacher(x_teacher)
            ema_logit = autograd.Variable(predict_teacher.detach().data, requires_grad=False)
            consistency_loss =softmax_mse_loss(raw_logits_student,ema_logit)/min_batch_size
            consistency_weight=1
            # sum_loss+=consistency_weight*consistency_loss
            # globals_consistency_loss += consistency_loss.data[0]

        loss = loss_arc + loss_center + consistency_weight*consistency_loss
        loss.backward()
        optimizer.step()
        optimizer_centloss.step()

        alpha = min(1 - 1 / (globals_step + 1), 0.99)
        update_ema_variables(net_student, net_teacher, alpha)

        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_center.parameters():
            param.grad.data *= (1. / 0.03)

        

        total_iters += 1
        alpha = min(1 - 1 / (total_iters), 0.99)
        update_ema_variables(net_student, net_teacher, alpha)

        if total_iters % 100 == 0:
            # current training accuracy
            _, preds = torch.max(output.data, 1)
            total = label.size(0)
            correct = (np.array(preds.cpu()) == np.array(label.data.cpu())).sum()
            time_cur = (time.time() - since) / 100
            since = time.time()

            for p in  optimizer.param_groups:
                lr = p['lr']
            print("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                .format(epoch, args.epoch-1, total_iters, loss.item(), correct/total, time_cur, lr))
            with open(train_logging_file, 'a') as f:
                f.write("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                .format(epoch, args.epoch-1, total_iters, loss.item(), correct/total, time_cur, lr)+'\n')
                f.close()

        # save model
        if total_iters % 5000 == 0:
            torch.save({
                'iters': total_iters,
                'net_state_dict': net_student.state_dict()},
                os.path.join(save_dir, 'Iter_%06d_student_model.ckpt' % total_iters))
            torch.save({
                'iters': total_iters,
                'net_state_dict': net_teacher.state_dict()},
                os.path.join(save_dir, 'Iter_%06d_teacher_model.ckpt' % total_iters))
            torch.save({
                'iters': total_iters,
                'net_state_dict': margin.state_dict()},
                os.path.join(save_dir, 'margin.ckpt'))

        # evaluate accuracy
        if total_iters % 5000 == 0:
            net_student.eval()
            # evaluate student
            for phase in ['privacy','LFW', 'CFP_FP', 'AgeDB30']:
                featureLs, featureRs = getFeature(net_student, dataloaders[phase], device, flip = args.flip)
                ACCs, threshold = evaluation_10_fold(featureLs, featureRs, dataset[phase], method = args.method)
                print('student: Epoch {}/{}，{} average acc:{:.4f} average threshold:{:.4f}'
                   .format(epoch, args.epoch-1, phase, np.mean(ACCs) * 100, np.mean(threshold)))

                with open(test_logging_file, 'a') as f:
                    f.write('student: Epoch {}/{}， {} average acc:{:.4f} average threshold:{:.4f}'
                         .format(epoch, args.epoch-1, phase, np.mean(ACCs) * 100, np.mean(threshold))+'\n')
                f.close()
                net_student.train()
            # featureLs=[], featureRs=[]
            # evaluate teacher 
            for phase in ['privacy','LFW', 'CFP_FP', 'AgeDB30']:
                featureLs_teacher, featureRs_teacher = getFeature(net_teacher, dataloaders[phase], device, flip = args.flip)
                ACCs_teacher, threshold_teacher = evaluation_10_fold(featureLs_teacher, featureRs_teacher, dataset[phase], method = args.method)
                print('teacher: Epoch {}/{}，{} average acc:{:.4f} average threshold:{:.4f}'
                   .format(epoch, args.epoch-1, phase, np.mean(ACCs_teacher) * 100, np.mean(threshold_teacher)))

                with open(test_logging_file, 'a') as f:
                    f.write('teacher: Epoch {}/{}， {} average acc:{:.4f} average threshold:{:.4f}'
                         .format(epoch, args.epoch-1, phase, np.mean(ACCs_teacher) * 100, np.mean(threshold_teacher))+'\n')
                f.close()

time_elapsed = time.time() - start  
print('Finally Best Accuracy: LFW: {:.4f} in iters: {}, CFP_FP: {:.4f} in iters: {} and AgeDB-30: {:.4f} in iters: {}'.format(
    best_acc['LFW'], best_iters['LFW'], best_acc['CFP_FP'], best_iters['CFP_FP'], best_acc['AgeDB30'], best_iters['AgeDB30']))
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))



