import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed

sys.path.append("../")
#sys.path.append("../../")
from utils.utils import *
from utils import KD_loss
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
#from birealnet import birealnet18
# from model import birealnet18
# from model import ResNet18_XNOR

#from custom_reactnet import custom_react
from model import AresBNet12
from tqdm import tqdm


def adjust_learning_rate(optimizer, epoch):
    update_list = [160, 200, 260, 320]
#    update_list = [40, 50, 70, 80]
#{{{
#    update_list = [120, 200, 260, 320]
#    update_list = [150, 250, 350]
#    update_list = [120, 200, 260, 320]
#    update_list = [150, 250, 350]
#    update_list = [150, 250, 320]
#    update_list = [15, 25, 35]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2
    return
#}}}  


parser = argparse.ArgumentParser("birealnet")
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
#For ImageNet
#parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
#For CIFAR100
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
#For ImageNet
#parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--save', type=str, default='./custom_models/Aresbnet', help='path for saving trained models')
parser.add_argument('--data', type=str, default='~/img_data/cifar100', metavar='DIR', help='path to dataset')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 40)')
parser.add_argument('--outputfile', action='store', default='./log/result.out',
                    help='output file')
parser.add_argument('--fold', type=int, default=5, help='fold rate')

args = parser.parse_args()

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

#CLASSES = 1000
CLASSES = 100

if not os.path.exists('log'):
    os.mkdir('log')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('log/log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
#{{{
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled=True
    logging.info("args = %s", args)

    # load model
    #model = birealnet18()
    #model = ResNet18_XNOR()
    #model = custom_react()
    model = AresBNet12()
    logging.info(model)
    model = nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if p.ndimension() == 4 or pname=='classifier.0.weight' or pname == 'classifier.0.bias':
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = torch.optim.AdamW(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
            lr=args.learning_rate,)
    #HJKIM: SGD optimizer 
#    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/(args.fold*args.epochs)), last_epoch=-1)
    start_epoch = 0
    best_top1_acc= 0

    checkpoint_tar = os.path.join(args.save, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_tar):
        logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        start_epoch = checkpoint['epoch']
        best_top1_acc = checkpoint['best_top1_acc']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logging.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))

    # adjust the learning rate according to the checkpoint
    for epoch in range(start_epoch):
        scheduler.step()
        #HJKIM
        #adjust_learning_rate(optimizer, epoch)

# for CIFAR-100
    # Data
    print('==> Preparing data..')
    crop_scale = 0.08
    transform_train = transforms.Compose([
         transforms.RandomCrop(32, padding=4),
         #transforms.RandomResizedCrop(32, scale=(crop_scale, 1.0)),
         #Lighting(0.1),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
         transforms.RandomErasing(),
     ])
 
    transform_test = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
     ])

    data_1 = torchvision.datasets.CIFAR100(root=args.data, train=True,
                                             download=True, transform=transform_train)
    data_2 = torchvision.datasets.CIFAR100(root=args.data, train=False,
                                             download=True, transform=transform_train)

    dataset = torch.utils.data.ConcatDataset([data_1,data_2])




    fold = args.fold
    split_size = len(dataset)//fold

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:

        top_1_total = 0
        top_5_total = 0

        for idx in range(fold):
            train_idx = list(range(0, idx*split_size)) + list(range((idx+1)*split_size, len(dataset)))
            val_idx = list(range(idx*split_size,(idx+1)*split_size))
            trainset = torch.utils.data.Subset(dataset,train_idx)
            testset = torch.utils.data.Subset(dataset,val_idx)
            train_loader = torch.utils.data.DataLoader(trainset,
                                                       batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False, num_workers=args.workers, pin_memory=True)


            train_obj, train_top1_acc, train_top5_acc = train(epoch, train_loader, model, criterion_smooth, optimizer,
                                                          scheduler)
            valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args)
            top_1_total += int(valid_top1_acc)
            top_5_total += int(valid_top5_acc)
              
        top_1_total /= fold
        top_5_total /= fold
    

        is_best = False
        if top_1_total > best_top1_acc:
            best_top1_acc = top_1_total
            is_best = True

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.save)

        print(' * acc@1 {} acc@5 {}'.format(top_1_total, top_5_total),best_top1_acc)
        
        epoch += 1
    #training_time = (time.time() - start_t) / 36000
    training_time = (time.time() - start_t) / 3600 # unit: one second

    #}}}

def train(epoch, train_loader, model, criterion, optimizer, scheduler):
#def train(epoch, train_loader, model, criterion, optimizer, adjust_learning_rate):
#{{{
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    #adjust_learning_rate(optimizer,epoch)
    scheduler.step()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    pbar = tqdm(train_loader,leave=True,desc="train")
    for i, (images, target) in enumerate(pbar):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        # compute outputy
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #progress.display(i)
        pbar.set_postfix({str(epoch):str(top1)})


    return losses.avg, top1.avg, top5.avg
#}}}


def validate(epoch, val_loader, model, criterion, args):
#{{{
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()

        pbar = tqdm(val_loader, leave=True,desc="valid")
        for i, (images, target) in enumerate(pbar):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #progress.display(i)
            pbar.set_postfix({str(epoch): str(top1)})

        #print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
        #      .format(top1=top1, top5=top5))

        outputfile_handler =open(args.outputfile, 'a+')
        print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5), file=outputfile_handler)
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
