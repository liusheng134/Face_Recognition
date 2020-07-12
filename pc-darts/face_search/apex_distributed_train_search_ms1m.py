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
from architect import Architect
from arcface import ArcMarginProduct,Softmax,Arcface
from load_dataset import load_data,load_data_with_dali,load_data_with_MXNet_dali
from evaluate import l2_norm,getFeature,getAccuracy,getThreshold,evaluation_10_fold

import torch.distributed as dist

from apex import amp 
from apex.parallel import DistributedDataParallel

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--data', type=str, default='/tmp/cache/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.3, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='./checkpoints_ms1m/', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--begin', type=int, default=0, help='batch size')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

parser.add_argument('--tmp_data_dir', type=str, default='../cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')

torch.set_num_threads(4)

args = parser.parse_args()
print(args.local_rank)

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)
torch.backends.cudnn.benchmark = True



args.save = '{}search-{}'.format(args.save, args.note)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

 #data preparation, we random sample 10% and 2.5% from training set(each class) as train and val, respectively.
#Note that the data sampling can not use torch.utils.data.sampler.SubsetRandomSampler as imagenet is too large
CLASSES = 85742
MULTI_GPU = True
resume = False
fineturn = False

best_acc = {'LFW': 0.0, 'CFP_FP': 0.0, 'AgeDB30': 0.0}
best_iters = {'LFW': 0, 'CFP_FP': 0, 'AgeDB30': 0}

# 加载数据
data_loaders , dataset_sizes, dataset = load_data_with_MXNet_dali(args.batch_size,args)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)

    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)

    logging.info("args = %s", args)



    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # criterion = torch.nn.DataParallel(criterion)
    criterion = criterion.cuda()

    # PC-DARTS模型初始换
    model = Network(args.init_channels, CLASSES, args.layers, criterion)
    model = model.cuda()

    # Arcface
    margin = ArcMarginProduct(512, CLASSES,criterion)
    # margin = torch.nn.DataParallel(margin)
    margin = margin.cuda()

    # pc-darts和margin的优化器
    if MULTI_GPU:
        optimizer = torch.optim.SGD(
            [{'params': model.parameters(), 'weight_decay': args.weight_decay},
            {'params': margin.parameters(), 'weight_decay': args.weight_decay}],
            args.learning_rate,
            momentum=args.momentum)
        # 架构参数的优化器
        optimizer_a = torch.optim.Adam(model.arch_parameters(),
                lr=args.arch_learning_rate, betas=(0.5, 0.999),
                weight_decay=args.arch_weight_decay)
    else:
        optimizer = torch.optim.SGD(
            [{'params': model.parameters(), 'weight_decay': args.weight_decay},
            {'params': margin.parameters(), 'weight_decay': args.weight_decay}],
            args.learning_rate,
            momentum=args.momentum)
            # 架构参数的优化器
        optimizer_a = torch.optim.Adam(model.arch_parameters(),
                lr=args.arch_learning_rate, betas=(0.5, 0.999),
                weight_decay=args.arch_weight_decay)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    margin, optimizer_a = amp.initialize(margin, optimizer_a, opt_level="O1")

    if MULTI_GPU:
        # model = torch.nn.DataParallel(model)
        # margin = torch.nn.DataParallel(margin)
        model = DistributedDataParallel(model)
        margin = DistributedDataParallel(margin)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        # margin = torch.nn.parallel.DistributedDataParallel(margin, device_ids=[args.local_rank])
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


    if resume:
        checkpoint = torch.load('/data/face_recognition/PC-DARTS-master/checkpoints_ms1m/search-try-20200423-094408/PC-DARTS_FACE.pth.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        margin.load_state_dict(checkpoint['margin_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # optimizer_a.load_state_dict(checkpoint['optimizer_a'])
        start_epoch = checkpoint['epoch']
        optimizer.param_groups[0]['lr'] = 0.5
        lr = args.learning_rate
    else:
        checkpoint = torch.load('./best/weights.pt')
        model.load_state_dict(checkpoint)
        checkpoint1 = torch.load('./best/margin.pt')
        margin.load_state_dict(checkpoint1)

        lr = args.learning_rate
        start_epoch = 0
        # sys.exit()

    # 混合精度
    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)




    for epoch in range(start_epoch,args.epochs):
        scheduler.step()
        current_lr = scheduler.get_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, current_lr)

        if epoch < 5 and args.batch_size > 256 and not fineturn:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
            # print(optimizer)
        if MULTI_GPU:
            print('ccccccccccvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvcccccccccccccccc')
            genotype = model.module.genotype()
            logging.info('genotype = %s', genotype)
            arch_param = model.module.arch_parameters()
        else:
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)
            arch_param = model.arch_parameters()



        # training
        train_acc, train_obj = train_with_dali(data_loaders['train_dataset'], model,margin, optimizer, optimizer_a, criterion, lr,epoch)
        logging.info('Train_acc %f', train_acc)

        # validation
        if epoch >= args.begin:
            infer(data_loaders,dataset, model, criterion,epoch)

        # utils.save(model, os.path.join(args.save, 'weights.pt'))
        # utils.save(margin, os.path.join(args.save, 'margin_weights.pt'))

        state = {'epoch': epoch + 1,
                 'args': args,
                 'margin_state_dict':margin.state_dict(),
                 'model_state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'optimizer_a': optimizer_a.state_dict()}
        # filename = 'checkpoints/' + args.output_filename + '.pth.tar'
        filename = os.path.join(args.save, 'PC-DARTS_FACE')+'.pth.tar'
        torch.save(state, filename)


def train(train_queue, valid_queue, model, margin, optimizer, optimizer_a, criterion, lr,epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)

        if epoch >=args.begin:
            optimizer_a.zero_grad()
            embbeding = model(input_search)
            thetas = margin(embbeding,target_search)

            loss_a = criterion(thetas, target_search)
            loss_a.sum().backward()
            if MULTI_GPU:
                nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            else:
                nn.utils.clip_grad_norm_(model.arch_parameters(), args.grad_clip)

            optimizer_a.step()
        #architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        embbeding = model(input)
        thetas = margin(embbeding,target)

        loss = criterion(thetas, target)

        loss.backward()
        if MULTI_GPU:
            nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        # break

        prec1, prec5 = utils.accuracy(thetas, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        global data_loaders,dataset
        if step % 5000 == 0 and step > 0 :
            infer(data_loaders,dataset, model, criterion,epoch)


        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)
            if MULTI_GPU:
                genotype = model.module.genotype()
                logging.info('genotype = %s', genotype)
                arch_param = model.module.arch_parameters()
            else:
                genotype = model.genotype()
                logging.info('genotype = %s', genotype)
                arch_param = model.arch_parameters()


            # filename = 'checkpoints/' + args.output_filename + '.pth.tar'
            utils.save(model, os.path.join(args.save, 'weights.pt'))
            # checkpoint = torch.load(os.path.join(args.save, 'weights.pt'))
            utils.save(margin, os.path.join(args.save, 'margin.pt'))
            # checkpoint = torch.load(os.path.join(args.save, 'margin.pt'))
            # model.load_state_dict(checkpoint)


    return top1.avg, objs.avg

def train_with_dali(train_queue, model, margin, optimizer, optimizer_a, criterion, lr,epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for i, data  in enumerate(train_queue):
        
        step = i
       
        model.train()
        

        input = data[0]["data"].cuda(non_blocking=True)
        target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
        n = input.size(0)
        # get a random minibatch from the search queue with replacement

        # try:
        #     input_search, target_search = next(train_queue)
        # except:
        #     valid_queue_iter = iter(train_queue)
        #     input_search, target_search = next(train_queue)
        # input_search = input_search.cuda(non_blocking=True)
        # target_search = target_search.cuda(non_blocking=True)

        if epoch >=args.begin and step % 2 ==0:
            optimizer_a.zero_grad()
            embbeding = model(input)
            thetas = margin(embbeding,target)

            loss_a = criterion(thetas, target)

            if loss_a.item() > 2e5:  # try to rescue the gradient explosion
                print("\nOh My God ! \nLoss is abnormal, drop this batch !")
                continue
            
            with amp.scale_loss(loss_a, optimizer_a) as scaled_loss:
                scaled_loss.backward()

            # loss_a.sum().backward()
            # if MULTI_GPU:
            #     nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            # else:
            #     nn.utils.clip_grad_norm_(model.arch_parameters(), args.grad_clip)

            # optimizer_a.step()
        
        # architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
        else:
            optimizer.zero_grad()
            embbeding = model(input)
            thetas = margin(embbeding,target)

            loss = criterion(thetas, target)

            if loss.item() > 2e5:  # try to rescue the gradient explosion
                print("\nOh My God ! \nLoss is abnormal, drop this batch !")
                continue
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            # loss.backward()
            # if MULTI_GPU:
            #     nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
            # else:
            #     nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # optimizer.step()

            prec1, prec5 = utils.accuracy(thetas, target, topk=(1, 5))
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

        global data_loaders,dataset
        if step % 500 == 0 and step > 0 :
            infer(data_loaders,dataset, model, criterion,epoch)


        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)
            if MULTI_GPU:
                genotype = model.module.genotype()
                logging.info('genotype = %s', genotype)
                arch_param = model.module.arch_parameters()
            else:
                genotype = model.genotype()
                logging.info('genotype = %s', genotype)
                arch_param = model.arch_parameters()


            # filename = 'checkpoints/' + args.output_filename + '.pth.tar'
            # utils.save(model, os.path.join(args.save, 'weights.pt'))
            # checkpoint = torch.load(os.path.join(args.save, 'weights.pt'))
            # utils.save(margin, os.path.join(args.save, 'margin.pt'))
            # checkpoint = torch.load(os.path.join(args.save, 'margin.pt'))
            # model.load_state_dict(checkpoint)


    return top1.avg, objs.avg


def infer(data_loaders,dataset, model, criterion,epoch):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for phase in ['LFW', 'CFP_FP', 'AgeDB30']:
        featureLs, featureRs = getFeature(model, data_loaders[phase], DEVICE, flip = True)
        ACCs, threshold = evaluation_10_fold(featureLs, featureRs, dataset[phase])
        print('Epoch {}/{},{} average acc:{:.4f} average threshold:{:.4f}'
              .format(epoch, args.epochs, phase, np.mean(ACCs) * 100, np.mean(threshold)))
        if best_acc[phase] <= np.mean(ACCs) * 100:
            best_acc[phase] = np.mean(ACCs) * 100
            # best_iters[phase] = total_iters

        logging.info('Epoch {}/{}, {} average acc:{:.4f} average threshold:{:.4f}'
                    .format(epoch, args.epochs-1, phase, np.mean(ACCs) * 100, np.mean(threshold))+'\n')





if __name__ == '__main__':
    main()

