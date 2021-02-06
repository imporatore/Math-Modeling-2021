import numpy as np

import torch
import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils import TRAIN_PIC_DIR, TEST_PIC_DIR
from model.config import BATCH_SIZE

transform = transforms.Compose([transforms.RandomResizedCrop(200),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(.5, .5, .5),
                                                     std=(.5, .5, .5))])

trainData = dsets.ImageFolder(TRAIN_PIC_DIR, transform=transform)
testData = dsets.ImageFolder(TRAIN_PIC_DIR, transform=transform)

target = np.array(trainData.targets)
class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
samples_weight = np.array([(1. / class_sample_count)[t] for t in target])
samples_weight = torch.from_numpy(samples_weight).double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, sampler=sampler)
testLoader = DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)
