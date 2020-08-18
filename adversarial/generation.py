'''
This script is for adversarially perturbing an input image to maximize model classifications. Input images are mean of class, with gaussian blur applied.

To run:

    python generation.py --load-model=path/to/weights --load-model2=path/to/different/weights --image-dir=path/to/h5

    output: saves images for each model (default naming is: supervised and contrastive)

'''


import os
import sys
import cv2
import argparse
import torch as ch
import numpy as np
import matplotlib.pyplot as plt

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from scipy import stats
from tqdm import tqdm, tqdm_notebook
from robustness import model_utils, datasets
from robustness.tools.vis_tools import show_image_row, show_image_column
from robustness.tools.label_maps import CLASS_DICT
from user_constants import DATA_PATH_DICT

sys.path.insert(1, os.getcwd())
from dataset import CXRDataset
from models.models import CombineNet
from models.mit_model import resnet14_1
from utils.utils import *
from utils.loss import *

parser = argparse.ArgumentParser(description='SIMCLR-CLASSI')
parser.add_argument('--load-model', type=str, default=None,
                    help='Load model for feature extraction (default None)')
parser.add_argument('--load-model2', type=str, default=None,
                    help='Load model for feature extraction (default None)')
parser.add_argument('--load-classifier', type=str, default=None,
                    help='Load model for feature extraction (default None)')
parser.add_argument('--image-dir', type=str, default='/data3/cxr_data/512_dataset.h5',
                    help='Path to dataset (default: data')
parser.add_argument('--output_dir', type=str, default='classifier',
                    help='Path to dataset (default: data')
args = parser.parse_args()

# Constants
DATA = 'CXR' # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
BATCH_SIZE = 1
NUM_WORKERS = 1
NUM_CLASSES_VIS = 4

DATA_SHAPE = 512 # Image size (fixed for dataset)
REPRESENTATION_SIZE = 2048 # Size of representation vector (fixed for model)
#CLASSES = CLASS_DICT[DATA] # Class names for dataset
CLASSES = {-1: 'noise', 0: 'none', 1: 'mild', 2: 'moderate', 3: 'severe'}
NUM_CLASSES = len(CLASSES) - 1
NUM_CLASSES_VIS = min(NUM_CLASSES_VIS, NUM_CLASSES)
GRAIN = 4 if DATA != 'CIFAR' else 1

device = ch.device("cuda")

# Load dataset
dataset = CXRDataset(args.image_dir, ['TEST'])
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE,pin_memory=True)
data_iterator = enumerate(test_loader)

#arch = CombineNet()#.to(device)
arch = resnet14_1(add_softmax=False, output_channels=4, latent_dim=192)
#arch = nn.DataParallel(arch)
#arch.eval()
print("Parameters: {}".format(sum(p.numel() for p in arch.parameters())))

# Load model
model_kwargs = {
    'arch': arch,
    'dataset': dataset,
    'resume_path': args.load_model + "/best.ckpt"
}

model, _ = model_utils.make_and_restore_model(**model_kwargs)
model.eval()

def downsample(x, step=GRAIN):
    down = ch.zeros([len(x), 1, DATA_SHAPE//step, DATA_SHAPE//step])

    for i in range(0, DATA_SHAPE, step):
        for j in range(0, DATA_SHAPE, step):
            v = x[:, :, i:i+step, j:j+step].mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            ii, jj = i // step, j // step
            down[:, :, ii:ii+1, jj:jj+1] = v
    return down

def upsample(x, step=GRAIN):
    up = ch.zeros([len(x), 1, DATA_SHAPE, DATA_SHAPE])

    for i in range(0, DATA_SHAPE, step):
        for j in range(0, DATA_SHAPE, step):
            ii, jj = i // step, j // step
            up[:, :, i:i+step, j:j+step] = x[:, :, ii:ii+1, jj:jj+1]
    return up


# Get seed distribution (can be memory intensive to do all ImageNet classes at once)

im_test, targ_test = [], []
for _, (im, targ, _) in enumerate(test_loader):
    im_test.append(im)
    _, y = targ.max(dim=1)
    targ_test.append(y)
im_test, targ_test = ch.cat(im_test), ch.cat(targ_test)
print(im_test.shape)
print(targ_test.shape)

conditionals = []
'''
for i in tqdm(range(NUM_CLASSES_VIS)):
    imc = im_test[targ_test == i]
    down_flat = downsample(imc).view(len(imc), -1)
    mean = down_flat.mean(dim=0)
    down_flat = down_flat - mean.unsqueeze(dim=0)
    cov = down_flat.t() @ down_flat / len(imc)
    dist = MultivariateNormal(mean, covariance_matrix=cov+1e-4*ch.eye(1 * DATA_SHAPE//GRAIN * DATA_SHAPE//GRAIN))
    conditionals.append(dist)
'''
imc = im_test
down_flat = downsample(imc).view(len(imc), -1)
mean = down_flat.mean(dim=0)
down_flat = down_flat - mean.unsqueeze(dim=0)
cov = down_flat.t() @ down_flat / len(imc)
dist = MultivariateNormal(mean, covariance_matrix=cov+1e-4*ch.eye(1 * DATA_SHAPE//GRAIN * DATA_SHAPE//GRAIN))
dist = dist.sample().view(1, DATA_SHAPE//GRAIN, DATA_SHAPE//GRAIN).cpu().numpy()
conditionals.append(dist)
conditionals.append(dist)
conditionals.append(dist)
conditionals.append(dist)

# Visualize seeds
#img_seed = ch.stack([conditionals[i].sample().view(1, DATA_SHAPE//GRAIN, DATA_SHAPE//GRAIN) 
#                     for i in range(NUM_CLASSES_VIS)])
conditionals = [cv2.GaussianBlur(conditionals[i],(11,11), 15) 
                     for i in range(NUM_CLASSES_VIS)]
conditionals = [cv2.GaussianBlur(conditionals[i],(11,11), 15) 
                     for i in range(NUM_CLASSES_VIS)]
img_seed = ch.stack([torch.tensor(conditionals[i]) for i in range(NUM_CLASSES_VIS)])
print(img_seed[0].shape)
img_seed = ch.clamp(img_seed, min=0, max=1)
#show_image_row([img_seed.cpu()], tlist=[[f'Class {i}' for i in range(NUM_CLASSES_VIS)]])

''' noise '''
'''
img_seed = ch.stack([torch.tensor(np.random.normal(0, 1, (1, 512, 512))) for i in range(NUM_CLASSES_VIS)])
img_seed = ch.clamp(img_seed, min=0, max=1)
'''

for i, img in enumerate(img_seed):
    cv2.imwrite("adversarial/seed-{}.png".format(i), np.array(img.cpu().squeeze() * 255, dtype=np.uint8))

def generation_loss(mod, inp, targ):
    op = mod(inp)
    loss = ch.nn.CrossEntropyLoss(reduction='none')(op, targ)
    return loss, None

kwargs = {
        'custom_loss': generation_loss,
        'constraint':'2',
        'eps': 50,
        'step_size': 1,
        'iterations': 100,
        'targeted': True,
}

show_seed = False
for i in range(NUM_CLASSES_VIS):
    target_class = i * ch.ones((BATCH_SIZE, )) 
    #im_seed = ch.stack([conditionals[int(t)].sample().view(1, DATA_SHAPE//GRAIN, DATA_SHAPE//GRAIN) 
    #                    for t in target_class])
    im_seed = img_seed[i]
    print(im_seed.shape)
    
    im_seed = upsample(ch.clamp(im_seed, min=0, max=1))

    #im_seed = torch.unsqueeze(img_seed[i], 0).float()
    print(im_seed.shape)
    _, im_gen = model(im_seed, target_class.long(), make_adv=True, **kwargs)
    #if show_seed:
    #    show_image_row([im_seed.cpu()], [f'Seed ($x_0$)'], fontsize=18)
    #show_image_row([im_gen.detach().cpu()], 
    #               [CLASSES[int(t)].split(',')[0] for t in target_class], 
    #               fontsize=18)
    for j, img in enumerate(im_gen):
        cv2.imwrite("adversarial/supervised/output1-{}-{}.png".format(i,j), np.array(img.cpu().squeeze() * 255, dtype=np.uint8))

arch = CombineNet()
print("Parameters: {}".format(sum(p.numel() for p in arch.parameters())))

# Load model
model_kwargs = {
    'arch': arch,
    'dataset': dataset,
    'resume_path': args.load_model2 + "/best.ckpt"
}

model, _ = model_utils.make_and_restore_model(**model_kwargs)
model.eval()

kwargs = {
        'custom_loss': generation_loss,
        'constraint':'2',
        'eps': 50,
        'step_size': 1,
        'iterations': 100,
        'targeted': True,
}

show_seed = False
for i in range(NUM_CLASSES_VIS):
    target_class = i * ch.ones((BATCH_SIZE, )) 
    im_seed = ch.stack([conditionals[int(t)].sample().view(1, DATA_SHAPE//GRAIN, DATA_SHAPE//GRAIN) 
                        for t in target_class])
    
    im_seed = upsample(ch.clamp(im_seed, min=0, max=1))

    #im_seed = torch.unsqueeze(img_seed[i], 0).float()
    print(im_seed.shape)
    _, im_gen = model(im_seed, target_class.long(), make_adv=True, **kwargs)
    #if show_seed:
    #    show_image_row([im_seed.cpu()], [f'Seed ($x_0$)'], fontsize=18)
    #show_image_row([im_gen.detach().cpu()], 
    #               [CLASSES[int(t)].split(',')[0] for t in target_class], 
    #               fontsize=18)
    for j, img in enumerate(im_gen):
        cv2.imwrite("adversarial/contrastive/output1-{}-{}.png".format(i,j), np.array(img.cpu().squeeze() * 255, dtype=np.uint8))

exit()

show_seed = False
for i in range(5):
    target_class = ch.tensor(np.random.choice(range(NUM_CLASSES_VIS), (BATCH_SIZE,)))
    im_seed = ch.stack([conditionals[int(t)].sample().view(1, DATA_SHAPE//GRAIN, DATA_SHAPE//GRAIN) 
                        for t in target_class])
    
    im_seed = upsample(ch.clamp(im_seed, min=0, max=1))
    _, im_gen = model(im_seed, target_class.long(), make_adv=True, **kwargs)
    for j, img in enumerate(im_gen):
        cv2.imwrite("output2-{}-{}.png".format(i,j), np.array(img.cpu().squeeze() * 255, dtype=np.uint8))
    '''
    if show_seed:
        show_image_row([im_seed.cpu()], [f'Seed ($x_0$)'], fontsize=18)
    show_image_row([im_gen.detach().cpu()], 
                   tlist=[[CLASSES[int(t)].split(',')[0] for t in target_class]], 
                   fontsize=18)
    '''
