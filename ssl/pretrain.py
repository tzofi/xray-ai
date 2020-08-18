# -*- coding: utf-8 -*-
from comet_ml import Experiment
import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torchvision.datasets as dset
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils.opencv_functional as cv2f
import cv2
import itertools
import torch.utils.model_zoo as model_zoo
import math
import random
from math import floor, ceil

from parsers import perturb_parser as parser
from models.resnet import ResidualNet
#from dataset import CXRImageDataset_InMem as CXRImageDataset
from models.mit_model import resnet14_1, resnet14_16, resnet14_4, resnet10_16
from peter_dataset import SSLDataset

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

args = parser.get_args()

''' Verify weights directory exists, if not create it '''
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

pixel_shift = 56 #0.25 * args.image_resize
expanded_params = ((0, -1 * pixel_shift, pixel_shift),(0, -1 * pixel_shift, pixel_shift))

shift = np.cumsum([0] + [len(p) for p in expanded_params[:-1]]).tolist()
num_params = [len(expanded_params[i]) for i in range(len(expanded_params))]
n_p1, n_p2 = num_params[0], num_params[1]
output_channels = sum(num_params) + 4  # +4 due to four rotations

device = torch.device('cuda')
FOLDS = args.training_folds + args.validation_folds
for validation_fold in FOLDS:
    print("Training with fold {} held out for validation".format(validation_fold))
    if not os.path.isdir(args.output_dir + "/" + str(validation_fold)):
        os.makedirs(args.output_dir + "/" + str(validation_fold))

    training_folds = [f for f in FOLDS if f != validation_fold] 

    # Datasets
    train_dataset = SSLDataset(args.image_dir, training_folds, expanded_params)
    val_dataset = SSLDataset(args.image_dir, [validation_fold], expanded_params)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=True)

    # Create instance of a resnet model
    add_softmax = False
    if args.model_architecture == 'resnet18':
        model = ResidualNet('ImageNet', 18, output_channels, 'CBAM', add_softmax=False, latent_dim=args.latent_dim)
    if args.model_architecture == 'resnet14_1':
        model = resnet14_1(add_softmax=add_softmax, 
                                              output_channels=output_channels, latent_dim=args.latent_dim)
    if args.model_architecture == 'resnet14_16':
        model = resnet14_16(add_softmax=add_softmax, 
                                               output_channels=output_channels, latent_dim=args.latent_dim)
    if args.model_architecture == 'resnet14_4':
        model = resnet14_4(add_softmax=add_softmax, 
                                              output_channels=output_channels, latent_dim=args.latent_dim)
    if args.model_architecture == 'resnet10_16':
        model = resnet10_16(add_softmax=add_softmax, 
                                               output_channels=output_channels, latent_dim=args.latent_dim)
    model = model.to(device)
    model = nn.DataParallel(model)

    optimizer = torch.optim.SGD(
        model.parameters(), args.init_lr, momentum=args.perturb_momentum,
        weight_decay=args.perturb_decay, nesterov=True)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.num_train_epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.init_lr))


    # /////////////// Training ///////////////
    def train():
        model.train()  # enter train mode
        loss_avg = 0.0
        for i, (data, target, img_id) in enumerate(train_loader):
            data = data.view(-1, 1, args.image_resize, args.image_resize)
            target = target.view(data.size(0), -1)
            t1, t2, t3 = target[:, 0], target[:, 1], target[:, 2]
            data, t1, t2, t3 = data.to(device), t1.to(device), t2.to(device), t3.to(device)

            # forward
            x,_ = model(2 * data - 1)

            # backward
            optimizer.zero_grad()
            loss = (F.cross_entropy(x[:, :n_p1], t1) +
                    F.cross_entropy(x[:, n_p1:n_p1 + n_p2], t2) +
                    F.cross_entropy(x[:, n_p1 + n_p2:], t3)) / 3.
            loss.backward()
            optimizer.step()
            scheduler.step()

            print("Step [{}/{}],\t Loss: {}".format(i, len(train_loader), loss.item()))

            # exponential moving average
            loss_avg = loss_avg * 0.9 + float(loss) * 0.1

        return loss_avg


    def test():
        loss_avg = 0.0
        model.eval()
        with torch.no_grad():
            for data, target, img_id in val_loader:
                data = data.view(-1, 1, args.image_resize, args.image_resize)
                target = target.view(data.size(0), -1)
                t1, t2, t3 = target[:, 0], target[:, 1], target[:, 2]
                data, t1, t2, t3 = data.cuda(), t1.cuda(), t2.cuda(), t3.cuda()

                # forward
                x,_ = model(2 * data - 1)

                loss = (F.cross_entropy(x[:, :n_p1], t1) +
                        F.cross_entropy(x[:, n_p1:n_p1 + n_p2], t2) +
                        F.cross_entropy(x[:, n_p1 + n_p2:], t3)) / 3.

                # test loss average
                loss_avg += float(loss.data)

        return loss_avg / len(val_loader)

    print('Beginning Training\n')

    # Main loop
    best_loss = np.inf
    early_stopping = 0
    for epoch in range(0, args.num_train_epochs):

        begin_epoch = time.time()

        train_loss = train()
        val_loss = test()
        early_stopping += 1

        # Save model
        if val_loss < best_loss:
            torch.save(model.state_dict(), args.output_dir + "/" + str(validation_fold) + "/best.ckpt")
            best_loss = val_loss 
            early_stopping = 0

        # Show results
        print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f}'.format(
            (epoch + 1),
            int(time.time() - begin_epoch),
            train_loss,
            val_loss)
        )

        if early_stopping > args.early_stopping:
            print("Early stopping reached!")
            break
