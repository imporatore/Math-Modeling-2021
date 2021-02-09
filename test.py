
import os

import numpy as np
import matplotlib.image as mp
import matplotlib.pyplot as plt

from utils.config import DATA_DIR

#%%

img = mp.imread(os.path.join(DATA_DIR, 'map1.jpg')).copy()[3:-3, 3:-3, :]
img_ = mp.imread(os.path.join(DATA_DIR, 'map.jpg')).copy()[3:-3, 3:-3, :3]


def conv_(layer, size=10, alpha=.1, gamma=.5):
    pos_kernal = np.array([[1 / ((i - (size - 1) / 2) ** 2 + (j - (size - 1) / 2) ** 2 + 1) ** alpha
                            for i in range(2 * size + 1)] for j in range(2 * size + 1)])
    soft_x = lambda x, kernal: x * np.exp(x * gamma) / np.sum(np.exp(x * gamma)) * kernal
    new_layer = np.array([[np.sum(soft_x(layer[max(0, i - size): min(layer.shape[0], i + size + 1),
                                         max(0, j - size): min(layer.shape[0], j + size + 1)],
                                         pos_kernal[max(0, size - i):min(layer.shape[1] + size - i, 2 * size + 1),
                                         max(0, size - j): min(layer.shape[1] + size - j, 2 * size + 1)]))
                           for j in range(layer.shape[1])] for i in range(layer.shape[0])])
    return new_layer

def conv(fig, size=10, alpha=.1, gamma=.5):
    return np.stack([conv_(fig[:, :, _], size=size, alpha=alpha, gamma=gamma) for _ in range(3)], axis=-1)

if __name__ == "__main__":
    conv(img[:12, :12, :])
