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
import torch.utils.data as data
from enas.models.controller import Controller
from enas.models.shared_cnn import SharedCNN,separate_resnet_bn_paras,FocalLoss
from enas.utils.utils import AverageMeter, Logger
from enas.utils.cutout import Cutout

from enas.dataloader import LFW, CFP_FP, AgeDB30, CASIAWebFace, MS1M

parser = argparse.ArgumentParser(description='ENAS')

parser.add_argument('--search_for', default='macro', choices=['macro'])
parser.add_argument('--data_path', default='/media/ls/861A943E1A942CE51/lfw-align-128/', type=str)
parser.add_argument('--output_filename', default='ENAS', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', type=int, default=2)
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

LOSS = FocalLoss()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

test_logging_file = 'test_MS1M_logging.txt'

def load_data(batch_size, dataset = 'Faces_emore'):
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    root = 'E:\\dataset/LFW/lfw_align_112'
    file_list = 'E:\\dataset/LFW/pairs.txt'
    dataset_LFW = LFW(root, file_list, transform=transform)
    
    root = 'E:\\dataset/CFP-FP/CFP_FP_aligned_112'
    file_list = 'E:\\dataset/CFP-FP/cfp_fp_pair.txt'
    dataset_CFP_FP = CFP_FP(root, file_list, transform=transform)
        
    root = 'E:\\dataset/AgeDB-30/agedb30_align_112'
    file_list = 'E:\\dataset/AgeDB-30/agedb_30_pair.txt'
    dataset_AgeDB30 = AgeDB30(root, file_list, transform=transform)  
    
    if dataset == 'CASIA':
        
        root = 'data_set/CASIA_Webface_Image'
        file_list = 'data_set/CASIA_Webface_Image/webface_align_112.txt'
        dataset_train = CASIAWebFace(root, file_list, transform=transform)
        
    elif dataset == 'Faces_emore':

        root = 'E:\\faces_emore_images'
        file_list = 'E:\\faces_emore_images\\faces_emore_align_112.txt'
        dataset_train = MS1M(root, file_list, transform=transform) 
    
    else:
        raise NameError('no training data exist!')

    dataset_size = len(dataset_train)
    train_size = int(0.8 * dataset_size)
    valid_size = dataset_size - train_size
    train_subset, valid_subset = torch.utils.data.random_split(dataset_train, [train_size, valid_size])
    
    dataloaders = {'train_dataset': data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1),
                   'train_subset': data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1),
                   'valid_subset': data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1),
                   'LFW': data.DataLoader(dataset_LFW, batch_size=batch_size, shuffle=False, num_workers=1),
                   'CFP_FP': data.DataLoader(dataset_CFP_FP, batch_size=batch_size, shuffle=False, num_workers=1),
                   'AgeDB30': data.DataLoader(dataset_AgeDB30, batch_size=batch_size, shuffle=False, num_workers=1)}
    
    dataset = {'train_dataset': dataset_train,'train_subset': train_subset,'valid_subset': valid_subset,'LFW': dataset_LFW,
               'CFP_FP': dataset_CFP_FP, 'AgeDB30': dataset_AgeDB30}
    
    dataset_sizes = {'train': len(dataset_train),'train_subset':len(train_subset),'valid_subset':len(valid_subset),'LFW': len(dataset_LFW),
                     'CFP_FP': len(dataset_CFP_FP), 'AgeDB30': len(dataset_AgeDB30)}
    
    print('training and validation data loaded')
    
    return dataloaders, dataset_sizes, dataset


def train_shared_cnn(epoch,
                     controller,
                     shared_cnn,
                     data_loaders,
                     shared_cnn_optimizer,
                     fixed_arc=None):
    """Train shared_cnn by sampling architectures from the controller.

    Args:
        epoch: Current epoch.
        controller: Controller module that generates architectures to be trained.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        data_loaders: Dict containing data loaders.
        shared_cnn_optimizer: Optimizer for the shared_cnn.
        fixed_arc: Architecture to train, overrides the controller sample
        ...
    
    Returns: Nothing.
    """
    # global vis_win

    controller.eval()

    if fixed_arc is None:
        # Use a subset of the training set when searching for an arhcitecture
        train_loader = data_loaders['train_subset']
    else:
        # Use the full training set when training a fixed architecture
        train_loader = data_loaders['train_dataset']

    train_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    for i, (images, labels) in enumerate(train_loader):
    # for det in train_loader:
        start = time.time()
        # print()
        # images, labels = det[0].to(DEVICE), det[1].to(DEVICE)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if fixed_arc is None:
            with torch.no_grad():
                controller()  # perform forward pass to generate a new architecture
            sample_arc = controller.sample_arc
        else:
            sample_arc = fixed_arc

        shared_cnn.zero_grad()
        outputs = shared_cnn(images, sample_arc,labels)
        
        loss = criterion(outputs, labels)
        # print(images)
        # print(outputs)
        # print('00000000000000000000000000000000000')
        # print(labels)
        loss.backward()
        # import sys
        # sys.exit()

        grad_norm = torch.nn.utils.clip_grad_norm_(shared_cnn.parameters(), args.child_grad_bound)
        shared_cnn_optimizer.step()

        _, preds = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (np.array(preds.cpu()) == np.array(labels.data.cpu())).sum()
        # print('cccccccccc')
        # print(correct)
        train_acc = correct/total
        # train_acc = torch.mean((torch.max(outputs, 1)[1] == labels).type(torch.float))

        train_acc_meter.update(train_acc.item())
        loss_meter.update(loss.item())

        end = time.time()

        if (i) % args.log_every == 0:
            learning_rate = shared_cnn_optimizer.param_groups[0]['lr']
            display = 'epoch=' + str(epoch) + \
                      '\tch_step=' + str(i) + \
                      '\tloss=%.6f' % (loss_meter.val) + \
                      '\tlr=%.4f' % (learning_rate) + \
                      '\t|g|=%.4f' % (grad_norm) + \
                      '\tacc=%.4f' % (train_acc_meter.val) + \
                      '\ttime=%.2fit/s' % (1. / (end - start))
            print(display)

    # vis_win['shared_cnn_acc'] = vis.line(
    #     X=np.array([epoch]),
    #     Y=np.array([train_acc_meter.avg]),
    #     win=vis_win['shared_cnn_acc'],
    #     opts=dict(title='shared_cnn_acc', xlabel='Iteration', ylabel='Accuracy'),
    #     update='append' if epoch > 0 else None)

    # vis_win['shared_cnn_loss'] = vis.line(
    #     X=np.array([epoch]),
    #     Y=np.array([loss_meter.avg]),
    #     win=vis_win['shared_cnn_loss'],
    #     opts=dict(title='shared_cnn_loss', xlabel='Iteration', ylabel='Loss'),
    #     update='append' if epoch > 0 else None)

    controller.train()


def train_controller(epoch,
                     controller,
                     shared_cnn,
                     data_loaders,
                     controller_optimizer,
                     baseline=None):
    """Train controller to optimizer validation accuracy using REINFORCE.

    Args:
        epoch: Current epoch.
        controller: Controller module that generates architectures to be trained.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        data_loaders: Dict containing data loaders.
        controller_optimizer: Optimizer for the controller.
        baseline: The baseline score (i.e. average val_acc) from the previous epoch
    
    Returns: 
        baseline: The baseline score (i.e. average val_acc) for the current epoch

    For more stable training we perform weight updates using the average of
    many gradient estimates. controller_num_aggregate indicates how many samples
    we want to average over (default = 20). By default PyTorch will sum gradients
    each time .backward() is called (as long as an optimizer step is not taken),
    so each iteration we divide the loss by controller_num_aggregate to get the 
    average.

    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L270
    """
    print('Epoch ' + str(epoch) + ': Training controller')

    # global vis_win

    shared_cnn.eval()
    valid_loader = data_loaders['valid_subset']

    reward_meter = AverageMeter()
    baseline_meter = AverageMeter()
    val_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    controller.zero_grad()
    for i in range(args.controller_train_steps * args.controller_num_aggregate):
        start = time.time()
        images, labels = next(iter(valid_loader))
        images = images.cuda()
        labels = labels.cuda()

        controller()  # perform forward pass to generate a new architecture
        sample_arc = controller.sample_arc

        with torch.no_grad():
            pred,_ = shared_cnn(images, sample_arc)

        _, preds = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (np.array(preds.cpu()) == np.array(labels.data.cpu())).sum()
        val_acc = correct/total
        # val_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))

        # detach to make sure that gradients aren't backpropped through the reward
        reward = torch.tensor(val_acc.detach())
        reward += args.controller_entropy_weight * controller.sample_entropy

        if baseline is None:
            baseline = val_acc
        else:
            baseline -= (1 - args.controller_bl_dec) * (baseline - reward)
            # detach to make sure that gradients are not backpropped through the baseline
            baseline = baseline.detach()

        loss = -1 * controller.sample_log_prob * (reward - baseline)

        if args.controller_skip_weight is not None:
            loss += args.controller_skip_weight * controller.skip_penaltys

        reward_meter.update(reward.item())
        baseline_meter.update(baseline.item())
        val_acc_meter.update(val_acc.item())
        loss_meter.update(loss.item())

        # Average gradient over controller_num_aggregate samples
        loss = loss / args.controller_num_aggregate

        loss.backward(retain_graph=True)

        end = time.time()

        # Aggregate gradients for controller_num_aggregate iterationa, then update weights
        if (i + 1) % args.controller_num_aggregate == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), args.child_grad_bound)
            controller_optimizer.step()
            controller.zero_grad()

            if (i + 1) % (2 * args.controller_num_aggregate) == 0:
                learning_rate = controller_optimizer.param_groups[0]['lr']
                display = 'ctrl_step=' + str(i // args.controller_num_aggregate) + \
                          '\tloss=%.3f' % (loss_meter.val) + \
                          '\tent=%.2f' % (controller.sample_entropy.item()) + \
                          '\tlr=%.4f' % (learning_rate) + \
                          '\t|g|=%.4f' % (grad_norm) + \
                          '\tacc=%.4f' % (val_acc_meter.val) + \
                          '\tbl=%.2f' % (baseline_meter.val) + \
                          '\ttime=%.2fit/s' % (1. / (end - start))
                print(display)

    # vis_win['controller_reward'] = vis.line(
    #     X=np.column_stack([epoch] * 2),
    #     Y=np.column_stack([reward_meter.avg, baseline_meter.avg]),
    #     win=vis_win['controller_reward'],
    #     opts=dict(title='controller_reward', xlabel='Iteration', ylabel='Reward'),
    #     update='append' if epoch > 0 else None)

    # vis_win['controller_acc'] = vis.line(
    #     X=np.array([epoch]),
    #     Y=np.array([val_acc_meter.avg]),
    #     win=vis_win['controller_acc'],
    #     opts=dict(title='controller_acc', xlabel='Iteration', ylabel='Accuracy'),
    #     update='append' if epoch > 0 else None)

    # vis_win['controller_loss'] = vis.line(
    #     X=np.array([epoch]),
    #     Y=np.array([loss_meter.avg]),
    #     win=vis_win['controller_loss'],
    #     opts=dict(title='controller_loss', xlabel='Iteration', ylabel='Loss'),
    #     update='append' if epoch > 0 else None)

    shared_cnn.train()
    return baseline

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def getFeature(net,best_arc, dataloader, device, flip = True):
    featureLs = None
    featureRs = None 
    count = 0
    for det in dataloader:
        for i in range(len(det)):
            det[i] = det[i].to(device)
        count += det[0].size(0)
        #print('extracing deep features from the face pair {}...'.format(count))
    
        with torch.no_grad():
            res = [net(d,best_arc).data.cpu() for d in det]
            
        if flip:      
            featureL = l2_norm(res[0] + res[1])
            featureR = l2_norm(res[2] + res[3])
        else:
            featureL = res[0]
            featureR = res[2]
        
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = torch.cat((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = torch.cat((featureRs, featureR), 0)
        
    return featureLs, featureRs

def getAccuracy(scores, flags, threshold, method):
    if method == 'l2_distance':
        p = np.sum(scores[flags == 1] < threshold)
        n = np.sum(scores[flags == -1] > threshold)
    elif method == 'cos_distance':
        p = np.sum(scores[flags == 1] > threshold)
        n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)

def getThreshold(scores, flags, thrNum, method):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 3.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i], method)
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold

def evaluation_10_fold(featureL, featureR, dataset, method = 'l2_distance'):
    
    ### Evaluate the accuracy ###
    ACCs = np.zeros(10)
    threshold = np.zeros(10)
    fold = np.array(dataset.folds).reshape(1,-1)
    flags = np.array(dataset.flags).reshape(1,-1)
    featureLs = featureL.numpy()
    featureRs = featureR.numpy()

    for i in range(10):
        
        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)
        
        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        if method == 'l2_distance':
            scores = np.sum(np.power((featureLs - featureRs), 2), 1) # L2 distance
        elif method == 'cos_distance':
            scores = np.sum(np.multiply(featureLs, featureRs), 1) # cos distance
        
        threshold[i] = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000, method)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold[i], method)
        
    return ACCs, threshold

def evaluate_model(epoch, controller, shared_cnn, data_loaders, n_samples=10):
    """Print the validation and test accuracy for a controller and shared_cnn.

    Args:
        epoch: Current epoch.
        controller: Controller module that generates architectures to be trained.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        data_loaders: Dict containing data loaders.
        n_samples: Number of architectures to test when looking for the best one.
    
    Returns: Nothing.
    """

    controller.eval()
    shared_cnn.eval()

    print('Here are ' + str(n_samples) + ' architectures:')
    best_arc, _ = get_best_arc(controller, shared_cnn, data_loaders, n_samples, verbose=True)

    valid_loader = data_loaders['valid_subset']
    for phase in ['LFW', 'CFP_FP', 'AgeDB30']:
        featureLs, featureRs = getFeature(model, data_loaders[phase], DEVICE, flip = args.flip)
        ACCs, threshold = evaluation_10_fold(featureLs, featureRs, dataset[phase])
        print('Epoch {}/{},{} average acc:{:.4f} average threshold:{:.4f}'
              .format(epoch, args.epoch-1, phase, np.mean(ACCs) * 100, np.mean(threshold)))
        if best_acc[phase] <= np.mean(ACCs) * 100:
            best_acc[phase] = np.mean(ACCs) * 100
            best_iters[phase] = total_iters

        with open(test_logging_file, 'a') as f:
            f.write('Epoch {}/{}, {} average acc:{:.4f} average threshold:{:.4f}'
                    .format(epoch, args.epoch-1, phase, np.mean(ACCs) * 100, np.mean(threshold))+'\n')
        f.close()        

    valid_acc = get_eval_accuracy(valid_loader, shared_cnn, best_arc)
    # test_acc = get_eval_accuracy(test_loader, shared_cnn, best_arc)

    print('Epoch ' + str(epoch) + ': Eval')
    print('valid_accuracy: %.4f' % (valid_acc))
    # print('test_accuracy: %.4f' % (test_acc))

    controller.train()
    shared_cnn.train()


def get_best_arc(controller, shared_cnn, data_loaders, n_samples=10, verbose=False):
    """Evaluate several architectures and return the best performing one.

    Args:
        controller: Controller module that generates architectures to be trained.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        data_loaders: Dict containing data loaders.
        n_samples: Number of architectures to test when looking for the best one.
        verbose: If True, display the architecture and resulting validation accuracy.
    
    Returns:
        best_arc: The best performing architecture.
        best_vall_acc: Accuracy achieved on the best performing architecture.

    All architectures are evaluated on the same minibatch from the validation set.
    """

    controller.eval()
    shared_cnn.eval()

    valid_loader = data_loaders['valid_subset']

    images, labels = next(iter(valid_loader))
    images = images.to('cpu')
    labels = labels.to('cpu')

    arcs = []
    val_accs = []
    for i in range(n_samples):
        with torch.no_grad():
            controller()  # perform forward pass to generate a new architecture
        sample_arc = controller.sample_arc
        arcs.append(sample_arc)

        with torch.no_grad():
            pred,_ = shared_cnn(images, sample_arc)
        val_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
        val_accs.append(val_acc.item())

        if verbose:
            print_arc(sample_arc)
            print('val_acc=' + str(val_acc.item()))
            print('-' * 80)

    best_iter = np.argmax(val_accs)
    best_arc = arcs[best_iter]
    best_val_acc = val_accs[best_iter]

    controller.train()
    shared_cnn.train()
    return best_arc, best_val_acc


def get_eval_accuracy(loader, shared_cnn, sample_arc):
    """Evaluate a given architecture.

    Args:
        loader: A single data loader.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        sample_arc: The architecture to use for the evaluation.
    
    Returns:
        acc: Average accuracy.
    """
    total = 0.
    acc_sum = 0.
    for (images, labels) in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred,_ = shared_cnn(images, sample_arc)
        correct += (np.array(preds.cpu()) == np.array(labels.data.cpu())).sum()
        # acc_sum += torch.sum((torch.max(pred, 1)[1] == labels).type(torch.float))
        total += pred.shape[0]

    acc = correct / total
    return acc.item()


def print_arc(sample_arc):
    """Display a sample architecture in a readable format.
    
    Args: 
        sample_arc: The architecture to display.

    Returns: Nothing.
    """
    for key, value in sample_arc.items():
        if len(value) == 1:
            branch_type = value[0].cpu().numpy().tolist()
            print('[' + ' '.join(str(n) for n in branch_type) + ']')
        else:
            branch_type = value[0].cpu().numpy().tolist()
            skips = value[1].cpu().numpy().tolist()
            print('[' + ' '.join(str(n) for n in (branch_type + skips)) + ']')


def train_enas(start_epoch,
               controller,
               shared_cnn,
               data_loaders,
               shared_cnn_optimizer,
               controller_optimizer,
               shared_cnn_scheduler):
    """Perform architecture search by training a controller and shared_cnn.

    Args:
        start_epoch: Epoch to begin on.
        controller: Controller module that generates architectures to be trained.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        data_loaders: Dict containing data loaders.
        shared_cnn_optimizer: Optimizer for the shared_cnn.
        controller_optimizer: Optimizer for the controller.
        shared_cnn_scheduler: Learning rate schedular for shared_cnn_optimizer
    
    Returns: Nothing.
    """

    baseline = None
    for epoch in range(start_epoch, args.num_epochs):

        train_shared_cnn(epoch,
                         controller,
                         shared_cnn,
                         data_loaders,
                         shared_cnn_optimizer)

        baseline = train_controller(epoch,
                                    controller,
                                    shared_cnn,
                                    data_loaders,
                                    controller_optimizer,
                                    baseline)

        if epoch % args.eval_every_epochs == 0:
            evaluate_model(epoch, controller, shared_cnn, data_loaders)

        shared_cnn_scheduler.step(epoch)

        state = {'epoch': epoch + 1,
                 'args': args,
                 'shared_cnn_state_dict': shared_cnn.state_dict(),
                 'controller_state_dict': controller.state_dict(),
                 'shared_cnn_optimizer': shared_cnn_optimizer.state_dict(),
                 'controller_optimizer': controller_optimizer.state_dict()}
        filename = 'checkpoints/' + args.output_filename + '.pth.tar'
        torch.save(state, filename)


def train_fixed(start_epoch,
                controller,
                shared_cnn,
                data_loaders):
    """Train a fixed cnn architecture.

    Args:
        start_epoch: Epoch to begin on.
        controller: Controller module that generates architectures to be trained.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        data_loaders: Dict containing data loaders.
    
    Returns: Nothing.

    Given a fully trained controller and shared_cnn, we sample many architectures,
    and then train a new cnn from scratch using the best architecture we found. 
    We change the number of filters in the new cnn such that the final layer 
    has 512 channels.
    """

    best_arc, best_val_acc = get_best_arc(controller, shared_cnn, data_loaders, n_samples=100, verbose=False)
    print('Best architecture:')
    print_arc(best_arc)
    print('Validation accuracy: ' + str(best_val_acc))

    fixed_cnn = SharedCNN(num_layers=args.child_num_layers,
                          num_branches=args.child_num_branches,
                          out_filters=512 // 4,  # args.child_out_filters
                          keep_prob=args.child_keep_prob,
                          fixed_arc=best_arc)
    fixed_cnn = fixed_cnn.cuda()

    fixed_cnn_optimizer = torch.optim.SGD(params=fixed_cnn.parameters(),
                                          lr=args.child_lr_max,
                                          momentum=0.9,
                                          nesterov=True,
                                          weight_decay=args.child_l2_reg)

    fixed_cnn_scheduler = CosineAnnealingLR(optimizer=fixed_cnn_optimizer,
                                            T_max=args.child_lr_T,
                                            eta_min=args.child_lr_min)

    test_loader = data_loaders['test_dataset']

    for epoch in range(args.num_epochs):

        train_shared_cnn(epoch,
                         controller,  # not actually used in training the fixed_cnn
                         fixed_cnn,
                         data_loaders,
                         fixed_cnn_optimizer,
                         best_arc)

        if epoch % args.eval_every_epochs == 0:
            test_acc = get_eval_accuracy(test_loader, fixed_cnn, best_arc)
            print('Epoch ' + str(epoch) + ': Eval')
            print('test_accuracy: %.4f' % (test_acc))

        fixed_cnn_scheduler.step(epoch)

        state = {'epoch': epoch + 1,
                 'args': args,
                 'best_arc': best_arc,
                 'fixed_cnn_state_dict': shared_cnn.state_dict(),
                 'fixed_cnn_optimizer': fixed_cnn_optimizer.state_dict()}
        filename = 'checkpoints/' + args.output_filename + '_fixed.pth.tar'
        torch.save(state, filename)



def main():
    global args

    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.manual_seed(args.seed)

    if args.fixed_arc:
        sys.stdout = Logger(filename='logs/' + args.output_filename + '_fixed.log')
    else:
        sys.stdout = Logger(filename='logs/' + args.output_filename + '.log')

    print(args)

    data_loaders , dataset_sizes, dataset = load_data(args.batch_size)

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

    # margin = Arcface(embedding_size=512, classnum=int(dataset['train_dataset'].class_nums),  s=32., m=0.5).cuda()

    shared_cnn = SharedCNN(num_layers=args.child_num_layers,
                           num_branches=args.child_num_branches,
                           out_filters=args.child_out_filters,
                           keep_prob=args.child_keep_prob)
    shared_cnn = shared_cnn.cuda()
    


    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L218
    controller_optimizer = torch.optim.Adam(params=controller.parameters(),
                                            lr=args.controller_lr,
                                            betas=(0.0, 0.999),
                                            eps=1e-3)

    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L213
    shared_cnn_optimizer = torch.optim.SGD(params=shared_cnn.parameters(),
                                           lr=args.child_lr_max,
                                           momentum=0.9,
                                           nesterov=True,
                                           weight_decay=args.child_l2_reg)
    
    # https://github.com/melodyguan/enas/blob/master/src/utils.py#L154
    shared_cnn_scheduler = CosineAnnealingLR(optimizer=shared_cnn_optimizer,
                                             T_max=args.child_lr_T,
                                             eta_min=args.child_lr_min)

    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
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

    if not args.fixed_arc:
        train_enas(start_epoch,
                   controller,
                   shared_cnn,
                   data_loaders,
                   shared_cnn_optimizer,
                   controller_optimizer,
                   shared_cnn_scheduler)
    else:
        assert args.resume != '', 'A pretrained model should be used when training a fixed architecture.'
        train_fixed(start_epoch,
                    controller,
                    shared_cnn,
                    data_loaders)


if __name__ == "__main__":
    main()
