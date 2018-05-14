import os
import sys

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import *

from config import BATCH_SIZE, LEARNING_RATE, RESTORE_INTERRUPTED, NUM_CLASSES, MASKS
from config import CHECKPOINT_DIR
from config import EPOCH_NUM
from config import IMAGES
from config import PER_EPOCH_LOSS
from config import PER_ITER_IOU
from config import PER_ITER_LOSS
from config import RESIZE_TO
from config import VAL_EPOCH_BCE
from config import gpu_id
from models import UNet11
from utils.abstract import DualCompose
from utils.abstract import ImageOnly
from utils.dataset import CvprSegmentationDataset
from utils.dualcolor import *
from utils.dualcrop import DualRotatePadded, DualCrop
from utils.loss import dice_coeff
from utils.metrics import image_iou
from utils.util_transform import DualResize, DualToTensor
from visualization.watchers import VisdomValueWatcher

watch = VisdomValueWatcher()

print(gpu_id)
print(IMAGES)

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

train_transform = DualCompose([
    # DualResize((RESIZE_TO, RESIZE_TO)),
    DualCrop(),
    DualRotatePadded(30),
    DualToTensor(),
    ImageOnly(Normalize(rgb_mean, rgb_std))])
#    ImageOnly(RandomSaturation(-0.1,0.1)),
#    ImageOnly(RandomGamma(0.9,1.1))])

test_transform = DualCompose([
    DualResize((RESIZE_TO, RESIZE_TO)),
    DualToTensor(),
    ImageOnly(Normalize(rgb_mean, rgb_std))
])

dataset = CvprSegmentationDataset(IMAGES, MASKS, 'train',
                                  train_transform, test_transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=1)

train_len = len(dataset)
dataset.set_mode('val')
val_len = len(dataset)
dataset.set_mode('train')

print('config batch size:', BATCH_SIZE)

upsample = torch.nn.Upsample(size=(RESIZE_TO, RESIZE_TO))
sigmoid = torch.nn.Sigmoid()


def eval_net(net, dataset, gpu=False):
    tot = 0
    loss = 0
    criterion = nn.CrossEntropyLoss().cuda()
    for i, b in tqdm(enumerate(dataset)):
        X = b[0]
        y = b[1]

        X = torch.FloatTensor(X).unsqueeze(0)
        y = torch.ByteTensor(y).unsqueeze(0)

        if gpu:
            X = Variable(X).cuda()
            y = Variable(y).cuda()
        else:
            X = Variable(X, volatile=True)
            y = Variable(y, volatile=True)

        y_pred = net(X)
        y_pred = (F.sigmoid(y_pred) > 0.5).float()

        dice = dice_coeff(y_pred, y.float()).data[0]
        tot += dice

        loss += criterion(y_pred.view(-1).float(), y.float()).cpu().data[0]

    return tot / float(i), loss / float(i)


def make_one_hot(labels, C=NUM_CLASSES):
    print(labels.size())
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = Variable(target)

    return target


def train_net(net, epochs=5, batch_size=8, lr=0.1, cp=True, gpu=True):
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, train_len, val_len, str(cp), str(gpu)))

    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss().cuda()

    scheduler = ReduceLROnPlateau(optimizer, 'min')
    for epoch_num in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch_num + 1, epochs))

        epoch_loss = 0
        dataset.set_mode('train')
        for i, b in tqdm(enumerate(loader)):
            X = b[0].cuda()
            y = b[1].cuda()
            y.requires_grad = False

            probs = F.sigmoid(net(X))
            print(probs.size(), y.size())
            print(probs)
            print(y)
            loss = criterion(probs.float(), y)

            watch.display_every_iter(i, X, y, probs, watch.get_vis())

            epoch_loss += loss.cpu().item()

            iou_m = image_iou(y, probs)

            watch.add_value(PER_ITER_IOU, iou_m)
            watch.output(PER_ITER_IOU)
            watch.add_value(PER_ITER_LOSS, loss.cpu().item())
            watch.output(PER_ITER_LOSS)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        watch.add_value(PER_EPOCH_LOSS, epoch_loss / float(i))
        watch.output(PER_EPOCH_LOSS)
        print('Epoch finished ! Loss: {}'.format(epoch_loss / float(i)))

        dataset.set_mode('val')
        net.eval()
        val_dice, val_bce = eval_net(net, dataset, gpu)
        watch.add_value(VAL_EPOCH_BCE, val_bce)
        watch.output(VAL_EPOCH_BCE)

        scheduler.step(val_bce)
        net.train()

        print('Validation Dice Coeff: {}, bce: {}'.format(val_dice, val_bce))

        if cp and epoch_num % 5 == 0:
            torch.save(net.state_dict(),
                       CHECKPOINT_DIR + 'linknet_{}_loss{}.pth'.format(epoch_num + 1, loss.data[0]))

            print('Checkpoint {} saved !'.format(epoch_num + 1))


if __name__ == '__main__':
    print(NUM_CLASSES)
    net = UNet11(NUM_CLASSES).cuda().double()
    # net = LinkNet34(NUM_CLASSES).cuda()
    cudnn.benchmark = True

    # if os.path.exists(RESTORE_INTERRUPTED) and RESTORE_INTERRUPTED is not None:
    #     net.load_state_dict(torch.load(RESTORE_INTERRUPTED))
    #     print('Model loaded from {}'.format('interrupted.pth'))
    try:
        train_net(net, EPOCH_NUM, BATCH_SIZE, LEARNING_RATE,
                  gpu=True)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), RESTORE_INTERRUPTED)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
