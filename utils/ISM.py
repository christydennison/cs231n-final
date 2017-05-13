
m __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np


class ISM(data.Dataset):

    def __init__(self, root, train=True):
        self.root = root
        self.train = train  # training set or test set

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(os.path.join(root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        return img, target

    def __len__(self):
        if self.train:
            return 60000
        else:
            return 10000
