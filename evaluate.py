import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import *

import cv2
from config import NUM_CLASSES
from config import IMAGES
from config import RESIZE_TO
from config import RESTORE_FROM
from models import LinkNet34
from utils.abstract import DualCompose
from utils.abstract import ImageOnly
from utils.dataset import JsonSegmentationDataset
from utils.dualcolor import *
from utils.metrics import image_iou
from utils.util_transform import DualResize, DualToTensor
from visualization.watchers import VisdomValueWatcher

watch = VisdomValueWatcher()

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

test_transform = DualCompose([
    DualResize((RESIZE_TO, RESIZE_TO)),
    DualToTensor(),
    ImageOnly(Normalize(rgb_mean, rgb_std))
])

jds = JsonSegmentationDataset(IMAGES, '/home/ubuntu/workdir/lyan/Pytorch-UNet/jsons/test.json', test_transform)
loader = DataLoader(jds, batch_size=2, num_workers=4)
kernel = np.ones((3, 3), np.uint8)


def postprocess(mask):
    t = mask.copy()
    thresh = 0.4
    t[t > thresh] = 1
    t[t <= thresh] = 0
    t = cv2.erode(t, kernel, iterations=1)
    t = cv2.dilate(t, kernel, iterations=1)
    t = cv2.morphologyEx(t, cv2.MORPH_OPEN, kernel)
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel)
    return t


def eval_net(net):
    net.eval()
    print('evaluation started')

    avg_iou = 0
    avg_dice = 0

    for i, b in tqdm(enumerate(loader)):

        X = b[0]
        y = b[1]
        if torch.cuda.is_available():
            X = Variable(X).cuda()
            y = Variable(y).cuda()
        else:
            X = Variable(X)
            y = Variable(y)

        probs = net(X)
        probs = postprocess(probs.cpu().data.numpy().reshape((RESIZE_TO, RESIZE_TO)))
        gt = y.cpu().data.numpy().reshape((RESIZE_TO, RESIZE_TO))

        avg_iou += image_iou(probs, gt, NUM_CLASSES)

    print('avg_iou:', avg_iou / float(i))


if __name__ == '__main__':
    net = LinkNet34().cuda()
    net.load_state_dict(torch.load(RESTORE_FROM))
    print('Model loaded from {}'.format(RESTORE_FROM))

    eval_net(net)
