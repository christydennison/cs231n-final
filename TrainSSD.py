from __future__ import print_function
import os
import sys
module_path = os.path.abspath(os.path.join('../ssd_pytorch/'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data.voc_invasive import AnnotationTransform, VOCDetection, detection_collate
from data.config_invasive import VOCroot, v2, v1
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import time
from ScaleSquareTransform import ScaleSquare


class SSDSolver(object):
    def __init__(self, model, num_classes=2, **kwargs):
        ## kwargs
        self.model = model
        self.batch_size = kwargs.pop('batch_size', 16)
        self.visdom = kwargs.pop('visdom', False)
        self.gamma = kwargs.pop('gamma', False)
        self.cuda = kwargs.pop('cuda', True)
        self.weight_decay = kwargs.pop('weight_decay', 0.0005)
        self.momentum = kwargs.pop('momentum', 0.9)
        self.lr = kwargs.pop('lr', True)
        self.save_folder = kwargs.pop('save_folder', 'weights')
        self.version = kwargs.pop('version', 'v2')
        
        self.accum_batch_size = 32
        self.iter_size = self.accum_batch_size / self.batch_size
        self.max_iter = 120000
        self.stepvalues = (80000, 100000, 120000)
        self.ssd_dim = 300  # only support 300 now
        self.rgb_means = (104, 117, 123)  # only support voc now
        
        ## initializations
        self.model.extras.apply(weights_init)
        self.model.loc.apply(weights_init)
        self.model.conf.apply(weights_init)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)
        self.criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
        
        if self.cuda:
            self.model.cuda()
            cudnn.benchmark = True

    def train(self):
        self.model.train()
        # loss counters
        loc_loss = 0  # epoch
        conf_loss = 0
        epoch = 0
        print('Loading Dataset...')

        dataset = VOCDetection(VOCroot, target_transform=AnnotationTransform())
        epoch_size = len(dataset) // self.batch_size
        print('Training SSD on', dataset.name)
        step_index = 0
        if self.visdom:
            # initialize visdom loss plot
            lot = viz.line(
                X=torch.zeros((1,)).cpu(),
                Y=torch.zeros((1, 3)).cpu(),
                opts=dict(
                    xlabel='Iteration',
                    ylabel='Loss',
                    title='Current SSD Training Loss',
                    legend=['Loc Loss', 'Conf Loss', 'Loss']
                )
            )
            epoch_lot = viz.line(
                X=torch.zeros((1,)).cpu(),
                Y=torch.zeros((1, 3)).cpu(),
                opts=dict(
                    xlabel='Epoch',
                    ylabel='Loss',
                    title='Epoch SSD Training Loss',
                    legend=['Loc Loss', 'Conf Loss', 'Loss']
                )
            )
        for iteration in range(self.max_iter):
            if iteration % epoch_size == 0:
                # create batch iterator
                batch_iterator = iter(data.DataLoader(dataset, self.batch_size,
                                                      shuffle=True, collate_fn=detection_collate))
            if iteration in self.stepvalues:
                step_index += 1
                adjust_learning_rate(self.optimizer, self.gamma, step_index)
                if self.visdom:
                    viz.line(
                        X=torch.ones((1, 3)).cpu() * epoch,
                        Y=torch.Tensor([loc_loss, conf_loss,
                            loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                        win=epoch_lot,
                        update='append'
                    )
                # reset epoch loss counters
                loc_loss = 0
                conf_loss = 0
                epoch += 1

            # load train data
            images, targets = next(batch_iterator)
            # print(images)
            # print(targets)
            if self.cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda()) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno) for anno in targets]
            # forward
            t0 = time.time()
            out = self.model(images)
            # backprop
            self.optimizer.zero_grad()
            loss_l, loss_c = self.criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            self.optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.data[0]
            conf_loss += loss_c.data[0]
            if iteration % 10 == 0:
                print('Timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            if self.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * iteration,
                    Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
                        loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
                    win=lot,
                    update='append'
                )
                # hacky fencepost solution for 0th epoch plot
                if iteration == 0:
                    viz.line(
                        X=torch.zeros((1, 3)).cpu(),
                        Y=torch.Tensor([loc_loss, conf_loss,
                            loc_loss + conf_loss]).unsqueeze(0).cpu(),
                        win=epoch_lot,
                        update=True
                    )
            if iteration % 5000 == 0:
                torch.save(self.model.state_dict(), 'weights/ssd300_invasive_iter_' +
                           repr(iteration) + '.pth')
        torch.save(net.state_dict(), self.save_folder + '' + self.version + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()