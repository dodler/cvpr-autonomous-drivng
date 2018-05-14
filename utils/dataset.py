import json
import os
import os.path as osp

import random
import cv2
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import NUM_CLASSES
from config import RESIZE_TO


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

    def crop(self, img):
        x = random.randint(0, img.shape[0] - RESIZE_TO)
        y = random.randint(0, img.shape[1] - RESIZE_TO)
        return img[x:x + RESIZE_TO, y:y + RESIZE_TO, :]

    def bin(self, m):
        t = m.copy() // 1000
        res = np.zeros((m.shape[0], m.shape[1], NUM_CLASSES))
        for label in self.labels2pixels.keys():
            pix = self.labels2pixels[label]
            t[t == label] = pix
            where = np.where(t == pix)
            res[where[0], where[1], pix] = 1

        return self.crop(res)

    def bin(self, m, target):
        t = m.copy() // 1000
        t[t == target] = 1
        t[t != target] = 0

        return t

    def __getitem__(self, item):
        if self.mode == 'train':
            path = osp.join(self.img_path, self.train[item])
            mask_path = osp.join(self.mask_path, self.train_mask[item])
            tr = self.train_transform
        else:
            path = osp.join(self.img_path, self.test[item])
            mask_path = osp.join(self.mask_path, self.test_mask[item])
            tr = self.test_transform

        im = self.crop(cv2.imread(path))
        m = self.bin(imread(mask_path), target=33)
        return tr(im, m)
