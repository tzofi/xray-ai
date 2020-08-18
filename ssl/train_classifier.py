import sys
import os
import time
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchlars import LARS

sys.path.insert(1, os.getcwd())
from dataset import CXRDataset
from models.resnet import *
from utils.utils import *
from utils.loss import *

parser = argparse.ArgumentParser(description='SSL-CLASSI')

parser.add_argument('--load-model', type=str, default=None,
                    help='Load model for feature extraction (default None)')
parser.add_argument('--image-dir', type=str, default='/data3/cxr_data/512_dataset.h5',
                    help='Path to dataset (default: data')
parser.add_argument('--output_dir', type=str, default='ssl/weights_finetune_512',
                    help='Path to dataset (default: data')
parser.add_argument('--feature-size', type=int, default=256,
                    help='Feature output size (default: 128')
parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                    help='input training batch-size')
parser.add_argument('--early-stop', type=int, default=25,
                    help='Epochs until early stop')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of training epochs (default: 150)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3')
parser.add_argument("--decay-lr", default=1e-6, action="store", type=float,
                    help='Learning rate decay (default: 1e-6')

args = parser.parse_args()

# train validate
def train_validate(model, loader, optimizer, is_train, epoch):

    loss_func = nn.CrossEntropyLoss()

    if is_train:
        model.train()
    else:
        model.eval()
    desc = 'Train' if is_train else 'Validation'

    total_loss = 0
    total_acc = 0

    for batch_idx, (x, y, _) in enumerate(loader):
        batch_loss = 0
        batch_acc = 0

        x = x.to(device)
        y = y.to(device)

        # Classify features
        y_hat = model(x)

        _, y = y.max(dim=1)
        loss = loss_func(y_hat, y)

        if is_train:
            model.zero_grad()
            loss.backward()
            optimizer.step()

        # Reporting
        batch_loss = loss.item() / x.size(0)
        total_loss += loss.item()

        pred = y_hat.max(dim=1)[1]
        correct = pred.eq(y).sum().item()
        correct /= y.size(0)
        batch_acc = (correct * 100)
        total_acc += batch_acc

        print('{} Epoch: {}, Step: [{}/{}], Average Loss: {:.4f}, Average Acc: {:.4f}, Batch Loss: {:.4f} Batch Acc: {:.4f}'.format(
            desc, epoch, batch_idx + 1, len(loader), total_loss/(batch_idx+1), total_acc/(batch_idx + 1), batch_loss, batch_acc))

    return total_loss / (batch_idx + 1), total_acc / (batch_idx + 1)

def execute_graph(model, loader, optimizer, epoch):
    t_loss, t_acc = train_validate(model, loader[0], optimizer, True, epoch)
    v_loss, v_acc = train_validate(model, loader[1], optimizer, False, epoch)

    print('Epoch: {} Total Train loss {}, Acc: {}'.format(epoch, t_loss, t_acc))
    print('Epoch: {} Total Valid loss {}, Acc: {}'.format(epoch, v_loss, v_acc))

    return v_loss

device = torch.device("cuda")

''' Verify weights directory exists, if not create it '''
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

test_dataset = CXRDataset(args.image_dir, ['TEST'])
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,pin_memory=True)

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
    train_dataset = CXRDataset(args.image_dir, training_folds, do_transform=True)
    val_dataset = CXRDataset(args.image_dir, [validation_fold])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,drop_last=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=True,pin_memory=True)
    loaders = [train_loader, val_loader]

    print('Total number of training images: {}, validation images: {}.'.format(len(train_dataset), len(val_dataset)))

    model = CombineNet(output_dim=output_dim).to(device)
    model = nn.DataParallel(model)

    if args.load_model:
        checkpoint = torch.load(args.load_model + "/" + str(validation_fold) + "/best.ckpt")
        model_weights = OrderedDict()
        for k, v in checkpoint['model'].items():
            name = k[7:] # remove `module.`
            model_weights[name] = v
        model.module.feature_model.load_state_dict(model_weights)
        epoch = checkpoint['epoch']
        print('Loading model: {}, from epoch: {}'.format(args.load_model, epoch))
    else:
        print('Model: {} not found'.format(args.load_model))
    #for param in model.module.feature_model.learning_head.parameters():
    #    param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    # Main training loop
    best_loss = np.inf
    epochs_since_best = 0

    for epoch in range(args.epochs):
        v_loss = execute_graph(model, loaders, optimizer, epoch)

        if v_loss < best_loss:
            best_loss = v_loss
            epochs_since_best = 0
            print('Writing model checkpoint')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'val_loss': v_loss
            }
            torch.save(state, args.output_dir + "/" + str(validation_fold) + "/best.ckpt")

        epochs_since_best += 1
        if epochs_since_best >= args.early_stop:
            print("Early stopping reached.")
            break

    t_loss, t_acc = train_validate(model, test_loader, optimizer, False, 0)
    print("Test Loss: {}, Test Acc: {}".format(t_loss, t_acc))
