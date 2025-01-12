"""
prepare datasets 
"""
import argparse
from pickletools import optimize
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.datasets as datasets
from model import ClassiFilerNet, FeatureNet
import time
import torchvision.transforms as transforms
import numpy as np
import shutil
from match_dataset import MatchDataset

INPUT_PIXEL = 128
parser = argparse.ArgumentParser(description='PyTorch matchNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
CUDA_DEVICE = 1

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main():
    args = parser.parse_args()
    input_pixel = INPUT_PIXEL
    
    class preporcessing(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, img):

            return torch.FloatTensor((img - 128 )/160.0).unsqueeze(0)

        def __repr__(self):
            pass

#    dataset = datasets.PhotoTour(root="./data/phototour", name="liberty", train=False, download=True,
#        transform = transforms.Compose([
#        preporcessing(),
#        #transforms.Normalize(mean=0.4437, std=0.2019)
#         ]))
#    print(dataset.labels[1:10])
#    print(dataset.matches[1:10])


#    valdataset = datasets.PhotoTour(root="./data/phototour", name="notredame", train=False, download=True,
#        transform = transforms.Compose([
#        preporcessing()
#        ])
#    )
#    print(valdataset.data.shape)
    dataset = MatchDataset(root_a="./img/train/tempA/", root_b="./img/train/tempB/",
                           input_pixel=INPUT_PIXEL)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    valdataset = MatchDataset(root_a="./img/test/tempA/", root_b="./img/test/tempB/",
                              input_pixel=INPUT_PIXEL)

    val_loader = torch.utils.data.DataLoader(
        valdataset,
        args.batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True) 

    torch.cuda.set_device(CUDA_DEVICE)
    model = ClassiFilerNet("alexNet", input_pixel)
#    model = ClassiFilerNet("mobileNet")
    model = model.cuda(CUDA_DEVICE)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(CUDA_DEVICE)

#    optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                                momentum=args.momentum,
#                                weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bestAcc = 0.0
    for epoch in range(0, args.epochs):

        # train for one epoch
        train(model, train_loader, criterion, optimizer, epoch)
        adjust_learning_rate(optimizer, epoch, args)
        if epoch % 10 == 0:
            acc = evaluate(model, valLoader=val_loader)
            is_best = acc > bestAcc
            bestAcc = max(bestAcc, acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': "matchNet",
                'state_dict': model.state_dict(),
                'best_acc1': bestAcc,
                'optimizer': optimizer.state_dict(),
            }, is_best)
        if epoch % 30 == 0:
            dataset = MatchDataset(root_a="./img/train/tempA/", root_b="./img/train/tempB/",
                                   input_pixel=INPUT_PIXEL)
            train_loader = torch.utils.data.DataLoader(
                dataset,
                args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True)
        if epoch % 50 == 0:
            valdataset = MatchDataset(root_a="./img/test/tempA/", root_b="./img/test/tempB/",
                                      input_pixel=INPUT_PIXEL)
            val_loader = torch.utils.data.DataLoader(
                valdataset,
                args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate(model, valLoader):
    accuracy = AverageMeter()
    model.eval()
    
    with torch.no_grad():
        
        for i, (data1, data2, matches) in enumerate(valLoader):

            batch = data1.shape[0]
            data1 = data1.cuda(CUDA_DEVICE)
            data2 = data2.cuda(CUDA_DEVICE)
            matches = matches.numpy()

            out = model((data1, data2)).cpu().numpy()
#            print("out is ==========")
#            print(out)
            prediction = np.array([x[1] > 0.55 for x in out])
#            print(prediction)
#            print(matches)
            count = np.sum(prediction == matches)
            accuracy.update(count/batch)

        print("accuracy:{}".format(accuracy.avg))
    
    return accuracy.avg

def train(model, trainLoader, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    for i, (data1, data2, matches) in enumerate(trainLoader):
        # measure data loading time
        data_time.update(time.time() - end)

        #test
        #print("data1:{}".format(data1))

        data1 = data1.cuda(CUDA_DEVICE)
        data2 = data2.cuda(CUDA_DEVICE)
        matches = matches.cuda(CUDA_DEVICE)

        out = model((data1, data2))
#        print(data1.shape)
#        print(data2.shape)
#        print(matches.shape)
#        print(out.shape)

        #test
        #print("data shape:{}".format(data1.shape))
        #print("out.shape{}, matches.shape{}".format(out.shape, matches.shape))

        loss = criterion(out, matches)

        losses.update(loss.item(), data1.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #with torch.no_grad():
        #    model.input_2.load_state_dict(model.input_1.state_dict())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   .format(
                      epoch, i, len(trainLoader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

if __name__ == "__main__":
    main()