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
from dataset import CXRDataset
from models.models import *
from utils.utils import *
from utils.loss import *

parser = argparse.ArgumentParser(description='SIMCLR-CLASSI')

parser.add_argument('--uid', type=str, default='SimCLR-CLASSI',
                    help='Staging identifier (default: SimCLR-CLASSI)')
parser.add_argument('--load-model', type=str, default=None,
                    help='Load model for feature extraction (default None)')
parser.add_argument('--load-classifier', type=str, default=None,
                    help='Load model for feature extraction (default None)')

parser.add_argument('--dataset-name', type=str, default='CIFAR10',
                    help='Name of dataset (default: CIFAR10')
parser.add_argument('--image-dir', type=str, default='/home/tz28264/repos/xray/512_dataset.h5',
                    help='Path to dataset (default: data')
parser.add_argument('--output_dir', type=str, default='classifier',
                    help='Path to dataset (default: data')
parser.add_argument('--feature-size', type=int, default=256,
                    help='Feature output size (default: 128')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input training batch-size')
parser.add_argument('--early-stop', type=int, default=100,
                    help='Epochs until early stop')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of training epochs (default: 150)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3')
parser.add_argument("--decay-lr", default=1e-6, action="store", type=float,
                    help='Learning rate decay (default: 1e-6')
parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')

args = parser.parse_args()

# train validate
def train_validate(classifier_model, feature_model, loader, optimizer, is_train, epoch):

    loss_func = nn.CrossEntropyLoss()

    classifier_model.train() if is_train else classifier_model.eval()
    desc = 'Train' if is_train else 'Validation'

    total_loss = 0
    total_acc = 0

    for batch_idx, (x, y, _) in enumerate(loader):
        batch_loss = 0
        batch_acc = 0

        x = x.to(device)
        y = y.to(device)

        # Get features
        f_x, _ = feature_model(x)
        f_x = f_x.detach()

        # Classify features
        y_hat = classifier_model(f_x)

        _, y = y.max(dim=1)
        loss = loss_func(y_hat, y)

        if is_train:
            classifier_model.zero_grad()
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

def execute_graph(classifier_model, feature_model, loader, optimizer, epoch):
    t_loss, t_acc = train_validate(classifier_model, feature_model, loader[0], optimizer, True, epoch)
    v_loss, v_acc = train_validate(classifier_model, feature_model, loader[1], optimizer, False, epoch)

    print('Epoch: {} Total Train loss {}, Acc: {}'.format(epoch, t_loss, t_acc))
    print('Epoch: {} Total Valid loss {}, Acc: {}'.format(epoch, v_loss, v_acc))

    return v_loss

device = torch.device("cuda")

''' Verify weights directory exists, if not create it '''
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

test_dataset = CXRDataset(args.image_dir, ['TEST'])
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,pin_memory=True)

feature_model = resnet18_cifar(args.feature_size).to(device)
feature_model = nn.DataParallel(feature_model)
feature_model.eval()

if args.load_model:
	checkpoint = torch.load(args.load_model + "/best.ckpt")
	feature_model.load_state_dict(checkpoint['model'])
	epoch = checkpoint['epoch']
	print('Loading model: {}, from epoch: {}'.format(args.load_model, epoch))
else:
	print('Model: {} not found'.format(args.load_model))

classifier_model = SimpleNet().to(device)
classifier_model = torch.nn.DataParallel(classifier_model)
classifier_model.eval()

optimizer = optim.Adam(classifier_model.parameters(), lr=args.lr, weight_decay=args.decay_lr)

if args.load_classifier:
	checkpoint = torch.load(args.load_classifier + "/best.ckpt")
	classifier_model.load_state_dict(checkpoint['model'])
	epoch = checkpoint['epoch']
	print('Loading model: {}, from epoch: {}'.format(args.load_classifier, epoch))
else:
	print('Model: {} not found'.format(args.classifier_model))

t_loss, t_acc = train_validate(classifier_model, feature_model, test_loader, optimizer, False, 0)
print("Test Loss: {}, Test Acc: {}".format(t_loss, t_acc))
