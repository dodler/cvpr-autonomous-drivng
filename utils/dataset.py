import os
import os.path as osp
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np

class CvprSegmentationDataset(Dataset):
    def __init__(self, img_path, mask_path, mode, train_transform, test_transform):
        self.test_transform = test_transform
        self.train_transform = train_transform
        self.mode = mode
        self.img_path = img_path
        self.mask_path = mask_path
        self.imgs = os.listdir(img_path)
        self.masks = os.listdir(mask_path)

        self.train, self.test, self.train_mask, self.test_mask = train_test_split(self.imgs, self.masks)

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.train)
        else:
            return len(self.test)

    def __getitem__(self, item):
        if self.mode == 'train':
            path = osp.join(self.img_path,self.train[item])
            mask_path = osp.join(self.mask_path, self.train_mask[item])
            tr = self.train_transform
        else:
            path = osp.join(self.img_path,self.test[item])
            mask_path = osp.join(self.mask_path,self.test_mask[item])
            tr = self.test_transform

        return tr(cv2.imread(path).astype(np.double), cv2.imread(mask_path).astype(np.double))