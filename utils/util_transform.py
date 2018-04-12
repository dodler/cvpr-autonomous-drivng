import cv2
import torch


class DualToTensor(object):
    def __call__(self, img, mask):
        img_tensor = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
        return img_tensor, torch.from_numpy(mask)


class DualResize(object):
    def __init__(self, target_shape):
        self._target_shape = target_shape

    def __call__(self, img, mask):
        return cv2.resize(img, self._target_shape), \
               cv2.resize(mask, self._target_shape)
