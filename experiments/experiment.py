import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
from os import mkdir

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from utils.utils import accuracy,AverageMeter

class FMExperiment(object):
    def __init__(self, wideresnet, params):
        self.model = wideresnet
        self.params = params
        self.curr_device = None
        # optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.params.optim_lr,
                          momentum=self.params.optim_momentum, nesterov=self.params.used_nesterov)

        # !!!!!! scheduler for the optimizer without warmup;  not finished (need to calculate total_training_step)
        self.total_training_step = 100000
        self.scheduler = LambdaLR(optimizer=self.optimizer,
                                  lr_lambda=lambda current_step :
                                  math.cos(7 * math.pi * current_step / (16 * self.total_training_step)))

        #save log
        summary_logdir = os.path.join(self.params.log_path, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)

        # used Gpu or not
        self.used_gpu = self.params['used_gpu']
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.used_gpu else 'cpu')

    def forward(self, input):
        return self.model(input)

    def training_step(self):
        start = time.time()
        batch_time = AverageMeter('Time', ':6.3f')
        train_losses = AverageMeter('Loss', ':.4e')
        # turn on model training
        self.model.train()
        for batch_idx, (inputs_labelled, targets_labelled) in enumerate(self.labelled_loader):
            if self.used_gpu:
                inputs_labelled = inputs_labelled.to(device = self.device)
                targets_labelled = targets_labelled.to(device=self.device)
            # forward
            outputs_labelled = self.forward(inputs_labelled)
            # compute loss
            loss = F.cross_entropy(outputs_labelled, targets_labelled, reduction='mean')

            # compute gradient and backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # update recording
            train_losses.update(loss.item(), inputs_labelled.shape[0])
            batch_time.update(time.time() - start)

        return train_losses.avg

    def testing_step(self):
        start = time.time()
        batch_time = AverageMeter('Time', ':6.3f')
        test_losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_dataloader):
                self.model.eval()
                if self.used_gpu:
                    inputs = inputs.to(device=self.device)
                    targets = targets.to(device=self.device)
                # forward
                outputs = self.forward(inputs)
                # compute loss and accuracy
                loss = F.cross_entropy(outputs, targets)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                # update recording
                test_losses.update(loss.item(), inputs.shape[0])
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))
                batch_time.update(time.time() - start)
        return test_losses.avg,top1.avg,top5.avg

    def fitting(self):
        prev_lr = np.inf

        if self.params.resume:
            # optionally resume from a checkpoint
            start_epoch, best_acc = self.resume_model()
        else:
            start_epoch, best_acc = 0, 0.0

        for epoch_idx in range(start_epoch, self.params.epoch_n):
            # turn on training
            start = time.time()
            train_loss = self.training_step()
            end = time.time()
            print("epoch {}: use {} seconds".format(epoch_idx, end - start))

            cur_lr = self.optimizer.param_groups[0]['lr']
            if cur_lr != prev_lr:
                print('--- Optimizer learning rate changed from %.2e to %.2e ---' % (prev_lr, cur_lr))
                prev_lr = cur_lr
            # testing
            test_loss, top1_acc, top5_acc= self.testing_step()

            self.swriter.add_scalars('train/loss', {'train_loss': train_loss,'test_loss': test_loss} ,epoch_idx)
            self.swriter.add_scalars('test/accuracy', {'Top1': top1_acc,'Top5': top5_acc},epoch_idx)


    def labelled_loader(self, labelled_training_dataset):
        self.num_train = len(labelled_training_dataset)
        self.labelled_loader = DataLoader(labelled_training_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           drop_last=True)
        return

    def unlabelled_loader(self,unlabelled_training_dataset):
        self.num_valid = len(unlabelled_training_dataset)
        self.unlabelled_loader = DataLoader(unlabelled_training_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           drop_last=True)
        return

    def test_loader(self,test_dataset):
        self.num_test_imgs = len(test_dataset) # same as len(dataloader)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          drop_last=True)
        return

    def load_model(self, mdl_fname, cuda=False):
        if self.used_gpu:
            self.model.load_state_dict(torch.load(mdl_fname))
            self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(mdl_fname, map_location='cpu'))
        self.model.eval()

    def resume_model(self):
        """ optionally resume from a checkpoint
        Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262 """
        start_epoch = 0
        best_acc = 0.0
        if self.params.resume:
            if os.path.isfile(self.params.resume_checkpoints):
                print("=> loading checkpoint '{}'".format(self.params.resume_checkpoints))
                if self.used_gpu:
                    # Map model to be loaded to specified single gpu.
                    checkpoint = torch.load(self.params.resume_checkpoints, map_location=self.device)
                else:
                    checkpoint = torch.load(self.params.resume_checkpoints)
                start_epoch = checkpoint['epoch']
                best_acc = checkpoint['best_acc']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                print("=> loaded checkpoint '{}' (epoch {})".format(self.params.resume_checkpoints, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.params.resume_checkpoints))
        return start_epoch,best_acc

    def end_writer(self):
        self.swriter.close()
