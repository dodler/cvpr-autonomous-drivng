import numpy as np


def image_iou(predict_mask, mask, classes):
    IOU = []
    for i in range(classes):
        intersection = ((mask == i) & (predict_mask == i)).sum()
        if intersection == 0:
            IOU.append(0)
            continue
        union = ((mask == i) | (predict_mask == i)).sum()
        if union == 0:
            IOU.append(-1)
            continue
        IOU.append(intersection / union)
    return np.mean(np.array(IOU))
