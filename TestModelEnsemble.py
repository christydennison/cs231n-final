import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
import timeit
from torchvision import models, transforms
from KaggleImageFolder import KaggleImageFolder
import os
from ScaleSquareTransform import ScaleSquare
cwd = os.getcwd()
import csv

class TestModelEnsemble(object):
    def __init__(self, models, scaling=224, dtype = torch.cuda.FloatTensor, scheme='ave'):
        dset = KaggleImageFolder(os.path.join(cwd, 'dataset/test'), transforms.Compose([ScaleSquare(scaling),transforms.ToTensor()]))
        self.dset_loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False, num_workers=1)
        self.models = models
        self.dtype = dtype
        self.scheme = scheme

    def submit(self, name="test_ens_0.csv", message="submission_ens_0"):
        name_to_pred = self._compute_test_results()
        self._create_submission_file(name, name_to_pred)
        self.send_to_kaggle(name, message)
        print("Submission complete!")

    def send_to_kaggle(self, name, message):
        os.system('kg submit ' + name + ' -u cs231n2017 -p cs231n2017 -c invasive-species-monitoring -m "' + message + '"')

    def _create_submission_file(self, name, name_to_pred):
        with open(name, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['name','invasive'])
            for key, value in name_to_pred.items():
                writer.writerow([key, value])

    def _compute_test_results(self):
        for model in self.models:
            model.eval()
        num_models = len(self.models)

        name_to_pred = {}
        softmax_fn = nn.Softmax()
        for x, full_path in self.dset_loader:
            x_var = Variable(x.type(self.dtype), volatile=True)

            scores = []
            for model in self.models:
                scores.append(softmax_fn(model(x_var))[:,0:2])

            probs = None
            if self.scheme == 'ave':
                probs = torch.stack(scores).sum(0).squeeze(0)/num_models
            elif self.scheme == 'max':
                probs = torch.stack(scores).squeeze(0).max(0)[0].squeeze(0)
            
            path = os.path.basename(full_path[0]).split('.jpg')[0]
            name_to_pred[path] = probs.data.cpu()[0,0]

        return name_to_pred
