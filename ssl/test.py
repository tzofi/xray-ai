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

parser = argparse.ArgumentParser(description='SSL-Test')

parser.add_argument('--load-model', type=str, default=None,
                    help='Load model for feature extraction (default None)')
parser.add_argument('--image-dir', type=str, default='/data3/cxr_data/512_dataset.h5',
                    help='Path to dataset (default: data')
parser.add_argument('--output_dir', type=str, default='classifier',
                    help='Path to dataset (default: data')
parser.add_argument('--feature-size', type=int, default=256,
                    help='Feature output size (default: 128')
parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                    help='input training batch-size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3')
parser.add_argument("--decay-lr", default=1e-6, action="store", type=float,
                    help='Learning rate decay (default: 1e-6')
parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')

args = parser.parse_args()

# train validate
def train_validate(model, loader, optimizer, is_train, epoch):

    loss_func = nn.CrossEntropyLoss()

    model.train() if is_train else model.eval()
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

device = torch.device("cuda")

''' Verify weights directory exists, if not create it '''
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

test_dataset = CXRDataset(args.image_dir, ['TEST'])
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,pin_memory=True)

test_acc = []
FOLDS = [1,2,3,4,5] 
for validation_fold in FOLDS:
    print("Testing with fold {} held out for validation".format(validation_fold))

    model = CombineNet().to(device)
    model = nn.DataParallel(model)
    print("Parameters: {}".format(sum(p.numel() for p in model.parameters())))
    model.eval()

    if args.load_model:
        checkpoint = torch.load(args.load_model + "/" + str(validation_fold) + "/best.ckpt")
		#model_weights = OrderedDict()
		#for k, v in checkpoint['model'].items():
		#	name = k[7:] # remove `module.`
		#	model_weights[name] = v
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        print('Loading model: {}, from epoch: {}'.format(args.load_model, epoch))
    else:
        print('Model: {} not found'.format(args.load_model))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay_lr)

    t_loss, t_acc = train_validate(model, test_loader, optimizer, False, 0)
    print("Test Loss: {}, Test Acc: {}".format(t_loss, t_acc))
    test_acc.append(t_acc)

print("Test Acc Mean: {}".format(np.mean(test_acc))) 
print("Test Acc Std: {}".format(np.std(test_acc)))
