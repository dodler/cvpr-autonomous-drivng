import cv2
import numpy as np
import torch
from torch.autograd import Variable

from config import NUM_CLASSES


def make_one_hot(labels, C=NUM_CLASSES):
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
    one_hot = torch.FloatTensor(C, labels.size(1), labels.size(2)).zero_()
    print(one_hot.size())
    return one_hot.scatter_(2, labels.squeeze(), 1)


class DualToTensor(object):
    def __call__(self, img, mask):
        img_tensor = torch.from_numpy(img[:, :, (2, 1, 0)].astype(np.float)).permute(2, 0, 1)

        t = np.zeros((NUM_CLASSES, mask.shape[0], mask.shape[1]))
        t[mask] = 1

        return img_tensor,torch.LongTensor(t)


class DualResize(object):
    def __init__(self, target_shape):
        self._target_shape = target_shape

    def __call__(self, img, mask):
        return cv2.resize(img, self._target_shape), \
               cv2.resize(mask, self._target_shape)
