import numpy as np
NUM_CLASSES = 36

mask = np.ones((36,32,32))

def colorify(mask):

    res = np.zeros((1, mask.shape[1], mask.shape[2]))
    for i in range(NUM_CLASSES):
        res[0, np.where(mask[i, :, :] == 1)] = i

    return res.astype(np.float)


n = colorify(mask)
print(n.shape)