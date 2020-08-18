import sys
import os
import time
import argparse
import torch
import torch.optim as optim
import numpy as np

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchlars import LARS

sys.path.insert(1, os.getcwd())
from dataset import CXRSimCLRDataset as CXRDataset
from models.models import *
from utils.utils import *
from utils.loss import *

parser = argparse.ArgumentParser(description='SIMCLR')

parser.add_argument('--image_dir', type=str, default='/mnt/USB/512_full_dataset.h5',
                    help='Path to dataset (default: data')
parser.add_argument('--feature-size', type=int, default=256,
                    help='Feature output size (default: 128')
parser.add_argument('--batch_size', type=int, default=44, metavar='N',
                    help='input training batch-size')
parser.add_argument('--accumulation-steps', type=int, default=22, metavar='N',
                    help='Gradient accumulation steps (default: 4')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of training epochs (default: 150)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3')
parser.add_argument("--decay-lr", default=1e-6, action="store", type=float,
                    help='Learning rate decay (default: 1e-6')
parser.add_argument('--tau', default=0.5, type=float,
                    help='Tau temperature smoothing (default 0.5)')
parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')
parser.add_argument('--output_dir', type=str, default='simclr/weights_512_full_batch_968',
                    help='logging directory (default: runs)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='disables multi-gpu (default: False')
parser.add_argument('--load-model', type=str, default=None,
                    help='Load model to resume training for (default None)')
parser.add_argument('--device-id', type=int, default=0,
                    help='GPU device id (default: 0')

args = parser.parse_args()

# train validate
def train_validate(model, loader, optimizer, is_train, epoch):

    loss_func = contrastive_loss(tau=args.tau)

    if is_train:
        model.train()
        model.zero_grad()
    else:
        model.eval()

    desc = 'Train' if is_train else 'Validation'

    total_loss = 0.0

    for i, (x_i, x_j, _, _) in enumerate(loader):

        x_i = x_i.to(device, non_blocking=True)
        x_j = x_j.to(device, non_blocking=True)

        _, z_i = model(x_i)
        _, z_j = model(x_j)

        loss = loss_func(z_i, z_j)
        loss /= args.accumulation_steps

        loss.backward()

        if (i + 1) % args.accumulation_steps == 0 and is_train:
            optimizer.step()
            model.zero_grad()

        total_loss += loss.item()

        print('{} Epoch: {}, Step: [{}/{}], Average Loss: {:.4f}, Loss: {:.4f}'.format(desc, epoch, i+1, len(loader), total_loss/(i+1), loss.item()))

    return total_loss / (len(loader.dataset))


def execute_graph(model, loaders, optimizer, scheduler, epoch):
    t_loss = train_validate(model, loaders[0], optimizer, True, epoch)
    v_loss = train_validate(model, loaders[1], optimizer, False, epoch)

    print("Validation loss: {}".format(v_loss))

    scheduler.step(v_loss)

    return v_loss

device = torch.device("cuda")

''' Verify weights directory exists, if not create it '''
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

all_results = {}
FOLDS = ['TEST',1,2,3,4,5,6] 
for validation_fold in FOLDS:
    print("Training with fold {} held out for validation".format(validation_fold))
    if not os.path.isdir(args.output_dir + "/" + str(validation_fold)):
        os.makedirs(args.output_dir + "/" + str(validation_fold))

    training_folds = [f for f in FOLDS if f != validation_fold] 

    # Datasets
    train_dataset = CXRDataset(args.image_dir, training_folds)
    val_dataset = CXRDataset(args.image_dir, [validation_fold])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,drop_last=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=True,pin_memory=True)
    loaders = [train_loader, val_loader]

    print('Total number of training images: {}, validation images: {}.'.format(len(train_dataset), len(val_dataset)))

    model = resnet18_cifar(args.feature_size).to(device)
    model = torch.nn.DataParallel(model)

    base_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay_lr)
    optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
    scheduler = ExponentialLR(optimizer, gamma=args.decay_lr)

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
                'optimizer': optimizer.state_dict(),
                'base_optimizer': base_optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_loss': v_loss
            }
            torch.save(state, args.output_dir + "/" + str(validation_fold) + "/best.ckpt")

    print("Only training once. Done.")
    exit()
