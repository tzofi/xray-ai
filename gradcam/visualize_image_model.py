#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18
#
# Adapted by Ruizhi Liao 

from __future__ import print_function

import copy
import os.path as osp
import os
import sys
from pathlib import Path
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from PIL import Image

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader

from grad_cam import (
	BackPropagation,
	Deconvnet,
	GradCAM,
	GuidedBackPropagation,
	occlusion_sensitivity,
)

import time
import argparse
import torch.optim as optim

from collections import OrderedDict
from torch.optim.lr_scheduler import ExponentialLR
from torchlars import LARS

from dataset import CXRDataset
from models import CombineNet 
from utils import *
from loss import *

parser = argparse.ArgumentParser(description='SIMCLR-CLASSI')

parser.add_argument('--uid', type=str, default='SimCLR-CLASSI',
                    help='Staging identifier (default: SimCLR-CLASSI)')
parser.add_argument('--load-model', type=str, default=None,
                    help='Load model for feature extraction (default None)')
parser.add_argument('--alpha', type=int, default=0.3,
                    help='Alpha value for blending gradcam and raw image')
parser.add_argument('--load-classifier', type=str, default=None,
                    help='Load model for feature extraction (default None)')

parser.add_argument('--dataset-name', type=str, default='CIFAR10',
                    help='Name of dataset (default: CIFAR10')
parser.add_argument('--image-dir', type=str, default='/home/tz28264/data/512_dataset.h5',
                    help='Path to dataset (default: data')
parser.add_argument('--output_dir', type=str, default='classifier',
                    help='Path to dataset (default: data')
parser.add_argument('--feature-size', type=int, default=256,
                    help='Feature output size (default: 128')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input training batch-size')
parser.add_argument('--early-stop', type=int, default=100,
                    help='Epochs until early stop')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of training epochs (default: 150)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3')
parser.add_argument("--decay-lr", default=1e-6, action="store", type=float,
                    help='Learning rate decay (default: 1e-6')
parser.add_argument('--log-dir', type=str, default='gradcam_output',
                    help='logging directory (default: runs)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')

args = parser.parse_args()

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False

def get_device(cuda):
	cuda = cuda and torch.cuda.is_available()
	device = torch.device("cuda" if cuda else "cpu")
	if cuda:
		current_device = torch.cuda.current_device()
		print("Device:", torch.cuda.get_device_name(current_device))
	else:
		print("Device: CPU")
	return device


def load_images(image_paths):
	images = []
	raw_images = []
	print("Images:")
	for i, image_path in enumerate(image_paths):
		print("\t#{}: {}".format(i, image_path))
		image, raw_image = preprocess(image_path)
		images.append(image)
		raw_images.append(raw_image)
	return images, raw_images


def get_classtable():
	classes = []
	with open("/data/vision/polina/projects/chestxray/code_joint/grad-cam-pytorch/samples/synset_words.txt") as lines:
		for line in lines:
			line = line.strip().split(" ", 1)[1]
			line = line.split(", ", 1)[0].replace(" ", "_")
			classes.append(line)
	return classes


def preprocess(image_path):
	raw_image = cv2.imread(image_path)
	raw_image = cv2.resize(raw_image, (224,) * 2)
	image = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)(raw_image[..., ::-1].copy())
	return image, raw_image


def save_gradient(filename, gradient):
	gradient = gradient.cpu().numpy().transpose(1, 2, 0)
	gradient -= gradient.min()
	gradient /= gradient.max()
	gradient *= 255.0
	cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
	gcam = gcam.cpu().numpy()
	cmap = cm.jet_r(gcam)[..., :3] * 255.0
	if paper_cmap:
		alpha = gcam[..., None]
		gcam = alpha * cmap + (1 - alpha) * raw_image
	else:
		#gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
		gcam = cmap.astype(np.float)
	gcam = gcam.astype(np.uint8)
	cv2.imwrite(filename, gcam)
	#cv2.imwrite(filename, np.uint8(gcam))
	return gcam


def save_sensitivity(filename, maps):
	maps = maps.cpu().numpy()
	scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
	maps = maps / scale * 0.5
	maps += 0.5
	maps = cm.bwr_r(maps)[..., :3]
	maps = np.uint8(maps * 255.0)
	maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
	cv2.imwrite(filename, maps)


def save_xray(filename, image):
	image = np.squeeze(image)
	cm = plt.get_cmap('gray')
	image = cm(image)
	xray = Image.fromarray((image[:, :, :3] * 255).astype(np.uint8))
	xray.save(filename)
	return xray

def main():
	"""
	Visualize model responses given multiple images
	"""
	device = torch.device("cuda")

	test_dataset = CXRDataset(args.image_dir, ['TEST'])
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

	model = CombineNet().to(device)
	model = nn.DataParallel(model)
	print("Parameters: {}".format(sum(p.numel() for p in model.parameters())))
	model.eval()


	if args.load_model:
		checkpoint = torch.load(args.load_model + "/best.ckpt")
		#model_weights = OrderedDict()
		#for k, v in checkpoint['model'].items():
		#	name = k[7:] # remove `module.`
		#	model_weights[name] = v
		model.load_state_dict(checkpoint['model'])
		epoch = checkpoint['epoch']
		print('Loading model: {}, from epoch: {}'.format(args.load_model, epoch))
	else:
		print('Model: {} not found'.format(args.load_model))

	# =========================================================================
	print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

	gcam = GradCAM(model=model)
	target_layer = 'module.feature_model.layer_block.3.1.conv2'
	#print(model.module.keys())
	print(model.module.feature_model.layer_block[3][1].conv2)

	epoch_iterator = tqdm(test_loader, desc="Iteration")
	for i, batch in enumerate(epoch_iterator, 0):
		raw_images, _, _ = batch
		batch_device = tuple(t.to(device, non_blocking=True) for t in batch)
		images, labels, img_ids = batch_device

		if (labels.cpu().detach().numpy() == np.array([0,0,0,1])).all():
			probs, ids  = gcam.forward(images)
			gcam.backward(ids=ids[:, [0]])
			regions = gcam.generate(target_layer=target_layer)

			for j in range(args.batch_size):
				gcam_im = save_gradcam(
					filename=osp.join(
						args.output_dir,
						"{}-gradcam-{}_severe_{}.png".format(j, target_layer, i),
						),
					gcam=regions[j, 0],
					raw_image=images[j],
					)
				img_path=osp.join(
						args.output_dir,
						"{}-image-{}_severe_{}.png".format(j, target_layer, i),
						)
				raw_image = np.array(raw_images[j])
				xray = save_xray(img_path, raw_image)
				'''
				# TODO: Blending code
				#cv2.imwrite(img_path, raw_images[j])

				img_path=osp.join(
						args.output_dir,
						"{}-blended-{}_severe_{}.png".format(j, target_layer, i),
						)
				beta = (1.0 - args.alpha)
				xray = np.array(xray)
				xray = xray.squeeze()
				xray = np.stack([xray, xray, xray],axis=2)
				print(xray.dtype)
				print(gcam.dtype)
				blended = cv2.addWeighted(xray, args.alpha, gcam, beta, 0.0)
				save_xray(img_path, blended)
				'''

	print('finished!')


if __name__ == '__main__':
	main()
