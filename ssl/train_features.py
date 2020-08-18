import sys
import os
import time
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchlars import LARS

sys.path.insert(1, os.getcwd())
from dataset import CXRSSLDataset as CXRDataset
from models.resnet import *
from utils.utils import *
from utils.loss import *

parser = argparse.ArgumentParser(description='SLL')

parser.add_argument('--image_dir', type=str, default='/data3/cxr_data/512_dataset.h5',
                    help='Path to dataset (default: data')
parser.add_argument('--feature-size', type=int, default=256,
                    help='Feature output size (default: 128')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input training batch-size')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of training epochs (default: 50)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 1e-3')
parser.add_argument("--decay-lr", default=1e-6, action="store", type=float,
                    help='Learning rate decay (default: 0.0005')
parser.add_argument("--momentum", default=0.9, action="store", type=float,
                    help='Momentum (default: 0.9')
parser.add_argument('--output_dir', type=str, default='ssl/weights_features_512',
                    help='logging directory (default: runs)')

args = parser.parse_args()

# train validate
def train_validate(model, loader, optimizer, scheduler, is_train, epoch):

    if is_train:
        model.train()
    else:
        model.eval()

    desc = 'Train' if is_train else 'Validation'

    total_loss = 0.0

    for i, (data, target, _) in enumerate(loader):
        data = data.view(-1, 1, 512, 512)
        target = target.view(data.size(0), -1)
        t1, t2, t3 = target[:, 0], target[:, 1], target[:, 2]
        data, t1, t2, t3 = data.to(device, non_blocking=True), t1.to(device, non_blocking=True), t2.to(device, non_blocking=True), t3.to(device, non_blocking=True)

        # forward
        x,_ = model(2 * data - 1)

        if is_train:
            model.zero_grad()

        loss = (F.cross_entropy(x[:, :n_p1], t1) +
                F.cross_entropy(x[:, n_p1:n_p1 + n_p2], t2) +
                F.cross_entropy(x[:, n_p1 + n_p2:], t3)) / 3.

        if is_train:
            loss.backward()
            optimizer.step()
            scheduler.step()

        total_loss += loss.item()

        print('{} Epoch: {}, Step: [{}/{}], Average Loss: {:.4f}, Loss: {:.4f}'.format(desc, epoch, i+1, len(loader), total_loss/(i+1), loss.item()))

    return total_loss / (len(loader.dataset))


def execute_graph(model, loaders, optimizer, scheduler, epoch):
    t_loss = train_validate(model, loaders[0], optimizer, scheduler, True, epoch)
    v_loss = train_validate(model, loaders[1], optimizer, scheduler, False, epoch)

    print("Validation loss: {}".format(v_loss))

    return v_loss

device = torch.device("cuda")

''' Verify weights directory exists, if not create it '''
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

expanded_params = ((0, -56, 56), (0, -56, 56))
num_params = [len(expanded_params[i]) for i in range(len(expanded_params))]
n_p1, n_p2 = num_params[0], num_params[1]
output_dim = sum(num_params) + 4  # +4 due to four rotations

all_results = {}
FOLDS = [1,2,3,4,5] 
for validation_fold in FOLDS:
    print("Training with fold {} held out for validation".format(validation_fold))
    if not os.path.isdir(args.output_dir + "/" + str(validation_fold)):
        os.makedirs(args.output_dir + "/" + str(validation_fold))

    training_folds = [f for f in FOLDS if f != validation_fold] 

    # Datasets
    train_dataset = CXRDataset(args.image_dir, training_folds, expanded_params)
    val_dataset = CXRDataset(args.image_dir, [validation_fold], expanded_params)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,drop_last=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=True,pin_memory=True)
    loaders = [train_loader, val_loader]

    print('Total number of training images: {}, validation images: {}.'.format(len(train_dataset), len(val_dataset)))

    model = ResidualNet('ImageNet', 18, output_dim, 'CBAM').to(device)
    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.SGD(
        model.parameters(), args.lr, momentum=args.momentum,
        weight_decay=args.decay_lr, nesterov=True)


    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.lr))

    # Main training loop
    best_loss = np.inf

    for epoch in range(args.epochs):
        v_loss = execute_graph(model, loaders, optimizer, scheduler, epoch)

        if v_loss < best_loss:
            best_loss = v_loss
            print('Writing model checkpoint')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'val_loss': v_loss
            }
            torch.save(state, args.output_dir + "/" + str(validation_fold) + "/best.ckpt")
