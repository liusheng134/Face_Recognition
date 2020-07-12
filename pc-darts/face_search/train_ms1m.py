import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from apex import amp
from torch.autograd import Variable
from model import NetworkImageNet as Network

from arcface import ArcMarginProduct,Softmax
from load_dataset import load_data,load_data_train_fix
from evaluate import l2_norm,getFeature,getAccuracy,getThreshold,evaluation_10_fold,getFeature_for_train_fix
from apex.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser("training imagenet")
parser.add_argument('--workers', type=int, default=12, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
parser.add_argument('--epochs', type=int, default=15, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='./checkpoints/', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PC_DARTS_FACE', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')
parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')

args, unparsed = parser.parse_known_args()

args.save = '{}eval-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 85742

MULTI_GPU = True
resume = True
fineturn = False

best_acc = {'LFW': 0.0, 'CFP_FP': 0.0, 'AgeDB30': 0.0} 
best_iters = {'LFW': 0, 'CFP_FP': 0, 'AgeDB30': 0}

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()


# 加载数据
data_loaders , dataset_sizes, dataset = load_data_train_fix(args.batch_size)
best_acc_top1 = 0
if not torch.cuda.is_available():
    logging.info('No GPU device available')
    sys.exit(1)
np.random.seed(args.seed)
cudnn.benchmark = True
torch.manual_seed(args.seed)
cudnn.enabled=True
torch.cuda.manual_seed(args.seed)
logging.info("args = %s", args)
logging.info("unparsed_args = %s", unparsed)
num_gpus = torch.cuda.device_count()   
genotype = eval("genotypes.%s" % args.arch)

def main():

    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------') 
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()
    # Arcface
    margin = ArcMarginProduct(512,CLASSES)
    margin = margin.cuda()

    if resume:
        print('cccccccccccccccccccccccccccccccc')
        checkpoint = torch.load('./margin.pt')
        margin.load_state_dict(checkpoint)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    if MULTI_GPU:
        optimizer = torch.optim.SGD(
            [{'params': model.parameters(), 'weight_decay': args.weight_decay},
            {'params': margin.parameters(), 'weight_decay': args.weight_decay}],
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
            )
    else:
        optimizer = torch.optim.SGD(
            [{'params': model.parameters(), 'weight_decay': args.weight_decay},
            {'params': margin.parameters(), 'weight_decay': args.weight_decay}],
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
            )


    if MULTI_GPU:
        model = torch.nn.DataParallel(model)
        margin = torch.nn.DataParallel(margin)

    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    
    best_acc_top5 = 0
    lr = args.learning_rate
    for epoch in range(args.epochs):
        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            print('Wrong lr type, exit')
            sys.exit(1)
        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)

        if MULTI_GPU:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    

        epoch_start = time.time()
        train_acc, train_obj = train(data_loaders['train_dataset'], model,margin, criterion, optimizer,epoch)
        logging.info('Train_acc: %f', train_acc)

        valid_acc_top1 = infer(data_loaders,dataset, model,margin,epoch)
        global best_acc_top1
        is_best = False
        if valid_acc_top1 > best_acc_top1:
                best_acc_top1 = valid_acc_top1
                is_best = True

        state={
            'epoch': epoch + 1,
            'model': model.module.state_dict(),
            'margin':margin.module.state_dict(),
            'best_acc_top1': best_acc_top1,
            'optimizer' : optimizer.state_dict(),
            }

        if is_best:
            filename = os.path.join('./', 'best_model.pth.tar')
            torch.save(state, filename)
            # torch.save(model.state_dict(), model_path)
            filename = os.path.join('./', 'checkpoint.pth.tar')
            torch.save(state, filename)
        else:
            filename = os.path.join('./', 'checkpoint.pth.tar')
            torch.save(state, filename)
        
        
               
        
def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs -  epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr        

def infer(data_loaders,dataset, model,margin,epoch):
    model.eval()

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

    return best_acc['LFW']

def infer1(data_loaders,dataset, model,margin,epoch):
    model.eval()

    return 0

def train(train_queue, model,margin, criterion, optimizer,epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        b_start = time.time()
        optimizer.zero_grad()
        logits = model(input)
        thetas = margin(logits,target)
        loss = criterion(thetas, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux

        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs', 
                                    step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)
        if step % 5000 == 0  :
            valid_acc_top1 = infer(data_loaders,dataset, model,margin,epoch)
            global best_acc_top1
            is_best = False
            if valid_acc_top1 > best_acc_top1:
                best_acc_top1 = valid_acc_top1
                is_best = True

            state={
                'epoch': epoch + 1,
                'model': model.module.state_dict(),
                'margin':margin.module.state_dict(),
                'best_acc_top1': best_acc_top1,
                'optimizer' : optimizer.state_dict(),
                }

            if is_best:
                filename = os.path.join('./', 'best_model.pth.tar')
                torch.save(state, filename)
                torch.save(model.state_dict(), model_path)
                filename = os.path.join('./', 'checkpoint.pth.tar')
                torch.save(state, filename)
            else:
                filename = os.path.join('./', 'checkpoint.pth.tar')
                torch.save(state, filename)

    return top1.avg, objs.avg



    ################################################

    


if __name__ == '__main__':
    main() 
