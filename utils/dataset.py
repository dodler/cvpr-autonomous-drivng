import os
import os.path as osp
import json
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np

import torch
from torch.autograd import Variable

from config import NUM_CLASSES


class CvprSegmentationDataset(Dataset):
    def __init__(self, img_path, mask_path, mode, train_transform, test_transform):
        self.test_transform = test_transform
        self.train_transform = train_transform
        self.mode = mode
        self.img_path = img_path
        self.mask_path = mask_path
        self.imgs = os.listdir(img_path)
        self.masks = os.listdir(mask_path)

        self.classes = json.load(open('jsons/labels.json', 'r'))
        self.labels2pixels = {}
        cnt = 0

        self.num_classes = len(self.classes)
        print(self.num_classes)

        for c in self.classes:
            self.labels2pixels[int(self.classes[c])] = cnt
            cnt += 1

        print(self.labels2pixels)

        self.train, self.test, self.train_mask, self.test_mask = train_test_split(self.imgs, self.masks)

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.train)
        else:
            return len(self.test)

    def binarize_one_hot(self, m):
        res = m.copy()
        print(res.shape)
        for label in self.labels2pixels.keys():
            res[res == label] = self.labels2pixels[label]

        print(np.unique(res))
        return res
        # t = np.transpose(np.eye(255)[res], (1, 2, 0))
        # print(t.shape)
        # return t[0:35, :, :]

    def __getitem__(self, item):
        if self.mode == 'train':
            path = osp.join(self.img_path, self.train[item])
            mask_path = osp.join(self.mask_path, self.train_mask[item])
            tr = self.train_transform
        else:
            path = osp.join(self.img_path, self.test[item])
            mask_path = osp.join(self.mask_path, self.test_mask[item])
            tr = self.test_transform

        im = cv2.imread(path)
        m = self.binarize_one_hot(cv2.imread(mask_path))
        m = m.reshape((m.shape[0], m.shape[1], 1))
        return tr(im, m)
