'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import utils.data as mt_data


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--train-subdir', type=str, default='train+val',
                        help='the subdirectory inside the data directory that contains the training data')
parser.add_argument('--eval-subdir', type=str, default='train+val',
                    help='the subdirectory inside the data directory that contains the evaluation data')
parser.add_argument('--labels', default='data-local/labels/cifar10/1000_balanced_labels/00.txt', type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')
parser.add_argument('--rand', default=False, type=bool, 
                        help='random select samples.')
parser.add_argument('--fuse', default=False, type=bool, 
                        help='fuse the score.')
parser.add_argument('--select-num', default=1000, type=int, 
                        help='num of samples selected each time.')
parser.add_argument('--epoch-num', default=5, type=int, 
                        help='num of epochs selected each time.')
parser.add_argument('--test-file', default='results/test.log', type=str, 
                        help='file to write acc.')
parser.add_argument('--interval', default=1, type=int,
                        help='interval of wave.')                                           

parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

# best_acc = 0  # best test accuracy

def main():
    # global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)


    # Data
    print('==> Preparing dataset %s' % args.dataset)
    # train and val
    train_transformation = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    data_dir = 'data-local/images/cifar/cifar10/by-image'
    num_classes = 10
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100
        data_dir = 'data-local/images/cifar/cifar100/by-image'

    dataset_config = {
    'train_transformation': train_transformation,
    'eval_transformation': eval_transformation,
    'datadir': data_dir,
    }
    trainloader, evalloader, unlabeled = create_data_loaders(**dataset_config, args=args)

    testset = dataloader(root='data-local/workdir/', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    if args.dataset == 'cifar10':
        title = 'cifar-10-' + args.arch
    else:
        title = 'cifar-100-' + args.arch

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        # best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Train Acc.', ])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    results = []
    if args.rand:
        print('select samples randomly...')
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        # test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        # if not args.rand and epoch >= args.epochs - args.epoch_num:  # last 5, interval = 1 epoch
        interval = args.interval # interval !=1 epochs
        if not args.rand and epoch in np.arange(args.epochs)[-1:-args.epoch_num*interval-1:-interval]: 
            results.append(get_p(evalloader, model, epoch, use_cuda))

        # append logger file
        # logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])
        logger.append([state['lr'], train_loss, train_acc])

        # save model
        # is_best = test_acc > best_acc
        # best_acc = max(test_acc, best_acc)

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                # 'acc': test_acc,
                # 'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, checkpoint=args.checkpoint)


    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    select_data(args.labels, results, unlabeled, args.select_num)
    # print('Best acc:')
    # print(best_acc)

def select_data(filename, results, unlabeled, select_num=1000):
    if args.dataset == 'cifar10':
        class_to_idx = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
            }
    else:
        class_to_idx = {
            'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4, 'bed': 5,
            'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9, 'bowl': 10,
            'boy': 11, 'bridge': 12, 'bus': 13, 'butterfly': 14, 'camel': 15,
            'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19, 'chair': 20,
            'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24, 'couch': 25,
            'crab': 26, 'crocodile': 27, 'cup': 28, 'dinosaur': 29, 'dolphin': 30,
            'elephant': 31, 'flatfish': 32, 'forest': 33, 'fox': 34, 'girl': 35,
            'hamster': 36, 'house': 37, 'kangaroo': 38, 'keyboard': 39, 'lamp': 40,
            'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44, 'lobster': 45,
            'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49, 'mouse': 50,
            'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55,
            'palm_tree': 56, 'pear': 57, 'pickup_truck': 58, 'pine_tree': 59, 'plain': 60,
            'plate': 61, 'poppy': 62, 'porcupine': 63, 'possum': 64, 'rabbit': 65,
            'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69, 'rose': 70,
            'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74, 'skunk': 75,
            'skyscraper': 76, 'snail': 77, 'snake': 78, 'spider': 79, 'squirrel': 80,
            'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84, 'tank': 85,
            'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89, 'train': 90,
            'trout': 91, 'tulip': 92, 'turtle': 93, 'wardrobe': 94, 'whale': 95,
            'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99
        }
    idx_to_class = {value:key for key, value in class_to_idx.items()}
    if len(results) == 0:  # random
        import random
        index = random.sample(range(len(unlabeled)), select_num)
    else:  # select according to precision
        predictions = np.array(results[-1])  # [num_img, num_cls]
        entropy =  - np.sum(predictions * np.log(predictions), axis=-1) # [num_img]
        results = np.array(results).transpose(1, 2, 0)  # [num_epoch, num_img, num_cls] -> [num_img, num_cls, num_epoch]
        final = []
        fusion = True
        if args.fuse: # fuse the stability and entropy
            for i, arr_img in enumerate(results):
                temp_var = [np.var(arr_epoch) for arr_epoch in arr_img] # var of a cls across epochs
                final.append(np.sum(temp_var) * entropy[i]) # fusion
            index = np.argsort(-np.array(final))[:select_num]
            with open('results/rank_fusion.txt', 'a') as f:
                f.write('final:' + str(np.array(final)[index][:50]) + '\n')
                f.close()
        else:
            index = np.argsort(-entropy)[:select_num]
            with open('results/rank_entropy.txt', 'a') as f:
                f.write('entropy:' + str(entropy[index][:50]) + '\n')
                f.close()

    # rewrite 00.txt
    with open(filename, 'a') as f:
        for i in index:
            f.write(os.path.basename(unlabeled[i][0]) + ' ' + idx_to_class[unlabeled[i][1]] + '\n')
        f.close()

    print('{} samples collected...'.format(select_num))
    return

def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):
    traindir = os.path.join(datadir, args.train_subdir)
    # evaldir = os.path.join(datadir, args.eval_subdir)

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation) # data-local/images/cifar/cifar10/by-image/train+val
    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = mt_data.relabel_dataset(dataset, labels)

    sampler = SubsetRandomSampler(labeled_idxs)
    batch_sampler = BatchSampler(sampler, args.train_batch, drop_last=True)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)
    eval_sampler = mt_data.SubsetSampler(unlabeled_idxs)
    eval_batch_sampler = BatchSampler(eval_sampler, 1, drop_last=False)
    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=eval_batch_sampler,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True)
    unlabel = [dataset.imgs[i] for i in unlabeled_idxs]

    return train_loader, eval_loader, unlabel

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    # global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    with open(args.test_file, 'a') as f:
        print('writing acc to:{}'.format(args.test_file))
        f.write('{} '.format(top1.avg))
        f.close()

    return (losses.avg, top1.avg)

def get_p(evalloader, model, epoch, use_cuda):
    # switch to evaluate mode
    model.eval()
    p_results = []
    bar = Bar('Evaluating', max=len(evalloader))
    for batch_idx, (inputs, targets) in enumerate(evalloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        outputs = nn.functional.softmax(outputs, dim=-1) # normalization
        p_results.append(outputs.cpu().data[0].numpy())

        bar.suffix = '({batch}/{size})'.format(batch=batch_idx+1, size=len(evalloader))
        bar.next()
    bar.finish()
    return p_results


def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
