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

class TestModel(object):
    def __init__(self, model, model_weights_path=None, scaling=224, dtype = torch.cuda.FloatTensor):
        dset = KaggleImageFolder(os.path.join(cwd, 'dataset/test'), transforms.Compose([ScaleSquare(scaling),transforms.ToTensor()]))
        self.dset_loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False, num_workers=1)
        if model_weights_path != None:
            model_save_dir = os.path.join(cwd, model_weights_path)
            model.load_state_dict(torch.load(model_save_dir))
        self.model = model
        self.dtype = dtype

    def submit(self, name="test0", message="submission0"):
        name_to_pred = self._compute_test_results()
        self._create_submission_file(name, name_to_pred)
        os.system('kg submit ' + name + ' -u cs231n2017 -p cs231n2017 -c invasive-species-monitoring -m "' + message + '"')
        print("Submission complete!")

    def _create_submission_file(self, name, name_to_pred):
        fname = name+'.csv'
        with open(fname, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['name','invasive'])
            for key, value in name_to_pred.items():
                writer.writerow([key, value])

    def _compute_test_results(self):
        self.model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
        name_to_pred = {}
        for x, full_path in self.dset_loader:
            x_var = Variable(x.type(self.dtype), volatile=True)
            scores = self.model(x_var)
            z, preds = scores.data.cpu().max(1)
            path = os.path.basename(full_path[0]).split('.jpg')[0]
            name_to_pred[path] = preds[0,0]
        return name_to_pred