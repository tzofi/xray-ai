'''
This file contains code for reading data using a NVIDIA DALI pipeline, which makes dataloading extremely fast.

The caveat to using this is that, at the time of writing, DALI does not support loading floating point data, only integer. Thus, for x-ray, this is not useable. For other data, such as natural images, this code could speed up data loading significantly.
'''


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import itertools

import csv
import os
import numpy as np
from math import floor, ceil
import scipy.ndimage as ndimage
from skimage import io
import cv2
import utils.opencv_functional as cv2f
import torchvision.transforms.functional as trnF

import torch.nn as nn
import torch.nn.functional as F

from parsers import parser as parser

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from random import shuffle

import cv2

from time import time


# Convert edema severity to ordinal encoding
def convert_to_ordinal(severity):
    if severity == 0:
        return [0,0,0]
    elif severity == 1:
        return [1,0,0]
    elif severity == 2:
        return [1,1,0]
    elif severity == 3:
        return [1,1,1]
    else:
        raise Exception("No other possibilities of ordinal labels are possible")

# Convert edema severity to one-hot encoding
def convert_to_onehot(severity):
    if severity == 0:
        return [1,0,0,0]
    elif severity == 1:
        return [0,1,0,0]
    elif severity == 2:
        return [0,0,1,0]
    elif severity == 3:
        return [0,0,0,1]
    else:
        raise Exception("No other possibilities of ordinal labels are possible")

# Load an .npy or .png image 
def load_image(img_path):
    if img_path[-3:] == 'npy':
        image = np.load(img_path)
        #_, image = cv2.imencode('.PNG', image)
    if img_path[-3:] == 'png':
        image = cv2.imread(img_path)
        image = image.astype(np.float32)
        image = image/np.max(image)
    return image

# Read the images and store them in the memory
def read_images(img_ids, root_dir):
    images = {}
    for img_id in list(img_ids.keys()):
        img_path = os.path.join(root_dir, img_id+'.npy')
        image = np.load(img_path)
        images[img_id] = image
    return images

class RandomTranslateCrop(object):
    """
    Translate, rotate and crop the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. 
        If int, square crop is made.
    """

    def __init__(self, output_size, shift_mean=0,
                 shift_std=200, rotation_mean=0, rotation_std=20):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.shift_mean = shift_mean
        self.shift_std = shift_std
        self.rotation_mean = rotation_mean
        self.rotation_std = rotation_std

    def __call__(self, image):

        image = self.__translate_2Dimage(image)
        #image = self.__rotate_2Dimage(image)
        h, w = image.shape[0:2]
        new_h, new_w = self.output_size

        if new_h>h or new_w>w:
            raise ValueError('This image needs to be padded!')

        top = floor((h - new_h) / 2)
        down = top + new_h
        left = floor((w - new_w) / 2)
        right = left + new_w
        
        return image[top:down, left:right]

    def __translate_2Dimage(self, image):
        'Translate 2D images as data augmentation'
        h, w = image.shape[0:2]
        h_output, w_output = self.output_size[0:2]

        # Generate random Gaussian numbers for image shift as data augmentation
        shift_h = int(np.random.normal(self.shift_mean, self.shift_std))
        shift_w = int(np.random.normal(self.shift_mean, self.shift_std))
        if abs(shift_h) > 2 * self.shift_std:
            shift_h = 0
        if abs(shift_w) > 2 * self.shift_std:
            shift_w = 0

        # Pad the 2D image
        pad_h_length = max(0, float(h_output - h))
        pad_h_length_1 = floor(pad_h_length / 2) + 4  # 4 is extra padding
        pad_h_length_2 = floor(pad_h_length / 2) + 4  # 4 is extra padding
        pad_h_length_1 = pad_h_length_1 + max(shift_h , 0)
        pad_h_length_2 = pad_h_length_2 + max(-shift_h , 0)

        pad_w_length = max(0, float(w_output - w))
        pad_w_length_1 = floor(pad_w_length / 2) + 4  # 4 is extra padding
        pad_w_length_2 = floor(pad_w_length / 2) + 4  # 4 is extra padding
        pad_w_length_1 = pad_w_length_1 + max(shift_w , 0)
        pad_w_length_2 = pad_w_length_2 + max(-shift_w , 0)

        image = np.pad(image, ((pad_h_length_1, pad_h_length_2), (pad_w_length_1, pad_w_length_2)),
                       'constant', constant_values=((0, 0), (0, 0)))

        return image

# Given a data split list (.csv), training folds and validation folds,
# return DICOM IDs and the associated labels for training and validation
def _split_tr_val(split_list_path, training_folds, validation_folds, use_test_data=True):
    """ Extracting finding labels """

    print('Data split list being used: ', split_list_path)

    train_labels = {}
    train_ids = {}
    val_labels = {}
    val_ids = {}
    test_labels = {}
    test_ids = {}


    with open(split_list_path, 'r') as train_label_file:
        train_label_file_reader = csv.reader(train_label_file)
        row = next(train_label_file_reader)
        for row in train_label_file_reader:
            if row[-1] != 'TEST':
                if int(row[-1]) in training_folds:
                    train_labels[row[2]] = [float(row[3])]
                    train_ids[row[2]] = row[1]
                if int(row[-1]) in validation_folds: #and not use_test_data:
                    val_labels[row[2]] = [float(row[3])]
                    val_ids[row[2]] = row[1]
            if row[-1] == 'TEST' and use_test_data:
                    test_labels[row[2]] = [float(row[3])]
                    test_ids[row[2]] = row[1]               

    print("Training and validation folds: ", training_folds, validation_folds)
    print("Total number of training labels: ", len(train_labels))
    print("Total number of training DICOM IDs: ", len(train_ids))
    print("Total number of validation labels: ", len(val_labels))
    print("Total number of validation DICOM IDs: ", len(val_ids))
    print("Total number of test labels: ", len(test_labels))
    print("Total number of test DICOM IDs: ", len(test_ids))

    return train_labels, train_ids, val_labels, val_ids, test_labels, test_ids

#Customizing dataset class for chest xray images
class CXRImageDataset(Dataset):
    
    def __init__(self, img_ids, labels, root_dir, 
    			 transform=None, image_format='png',
    			 encoding='ordinal'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.img_ids = img_ids
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.image_format = image_format
        self.encoding = encoding

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = list(self.img_ids.keys())[idx]
        img_path = os.path.join(self.root_dir,
                                img_id+'.'+self.image_format)
        image = load_image(img_path)
        print("max/min")
        print(np.max(image))
        print(np.min(image))
        image = cv2.resize(image, (1024,1024)) 
        #if self.transform:
        #    image = self.transform(image)
        image = image.reshape(1, image.shape[0], image.shape[1])
        
        label = self.labels[img_id]
        if self.encoding == 'ordinal':
        	label = convert_to_ordinal(label[0])
        if self.encoding == 'onehot':
        	label = convert_to_onehot(label[0])
        label = torch.tensor(label, dtype=torch.float32)

        sample = [image, label]

        return sample

#Customizing dataset class for chest xray images
class CXRImageIterator(object):
    def __init__(self, batch_size, device_id, num_gpus, img_ids, labels, root_dir, image_format='npy', encoding='onehot'):
        self.root_dir = root_dir
        self.image_format = image_format
        self.encoding = encoding

        self.files = []
        for img_id in list(img_ids.keys()):
            self.files.append([img_id, labels[img_id]])
        self.batch_size = batch_size
        # whole data set size
        self.data_set_len = len(self.files)

        # based on the device_id and total number of GPUs - world size
        # get proper shard
        self.files = self.files[self.data_set_len * device_id // num_gpus:
                                self.data_set_len * (device_id + 1) // num_gpus]
        self.n = len(self.files)

    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            raise StopIteration

        for _ in range(self.batch_size):
            filename, label = self.files[self.i]
            img_path = os.path.join(self.root_dir, filename + '.' + self.image_format)
            # This might need to be converted to np.uint8
            f = open(img_path, 'rb')
            buf = np.frombuffer(f.read(), dtype = np.uint8)
            batch.append(buf)
            #batch.append(np.frombuffer(f.read(), dtype = np.float32))
            #img = batch.append(load_image(img_path))

            if self.encoding == 'ordinal':
                    label = convert_to_ordinal(label[0])
            if self.encoding == 'onehot':
                    label = np.array(convert_to_onehot(label[0]), dtype=np.uint8)
            labels.append(label)

            self.i = (self.i + 1) % self.n

        return (batch, labels)

    @property
    def size(self,):
        return self.data_set_len

    next = __next__


class CXRImagePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, external_data):
        super(CXRImagePipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.GRAY)
        #self.norm = ops.Normalize(device="cpu")
        self.res = ops.Resize(device="gpu", resize_x=1024, resize_y=1024)
        self.cast = ops.Cast(device = "gpu",
                             dtype = types.FLOAT)
        self.external_data = external_data
        self.iterator = iter(self.external_data)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        images = self.decode(self.jpegs)
        #images = self.norm(images)
        #images = self.res(images)
        output = self.cast(images)
        return (output, self.labels)

    def iter_setup(self):
        try:
            (images, labels) = self.iterator.next()
            self.feed_input(self.jpegs, images)
            self.feed_input(self.labels, labels)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


if __name__ == "__main__":

    args = parser.get_args()
    train_labels, train_ids, val_labels, val_ids, test_labels, test_ids = _split_tr_val(args.data_split_path, 
                                                                            args.training_folds, 
                                                                            args.validation_folds)

    '''
    for f in list(train_ids.keys()):
        im = np.load(args.image_dir + "/" + f + ".npy")
        im = np.array(im, dtype=np.float32) * 255
        cv2.imwrite("/mnt/USB/png/" + f + ".png",im)
    for f in list(val_ids.keys()):
        im = np.load(args.image_dir + "/" + f + ".npy")
        im = np.array(im, dtype=np.float32) * 255
        cv2.imwrite("/mnt/USB/png/" + f + ".png",im)
    for f in list(test_ids.keys()):
        im = np.load(args.image_dir + "/" + f + ".npy")
        im = np.array(im, dtype=np.float32) * 255
        cv2.imwrite("/mnt/USB/png/" + f + ".png",im)
    exit()
    '''

    '''
    img_id = list(train_ids.keys())
    labels = []
    for im in img_id:
        labels.append([im, train_labels[im]])
    print(labels)
    '''

    from torch.utils.data import DataLoader
    train_transform = RandomTranslateCrop(args.image_resize)
    train_dataset = CXRImageDataset(train_ids, train_labels, args.image_dir,
                                  transform=train_transform, image_format=args.image_format,
                                  encoding=args.label_encoding)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, drop_last=True, num_workers=2)

    device = torch.device("cuda")
    times = []
    for epoch in range(1):
        for i, batch in enumerate(train_loader):
            if i == 100: break 
            if i != 0:
                stop = time()
                times.append(stop-start)
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            exit()
            inputs, labels = batch
            print("epoch: {}, iter {}".format(epoch, i))
            start = time()
    print(np.mean(times))
    exit()


    '''
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = '8'
    os.environ['RANK'] = '0'
    distributed = False
    local_rank = 0
    print("working")
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    print(distributed)

    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://')

    print("worked!")
    N_gpu = torch.distributed.get_world_size()
    print("N_gpus: " + str(N_gpu))
    exit()
    if distributed:
        torch.cuda.set_device(0)
        print("device set")
        torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=0)
        print("init done")
        #torch.distributed.init_process_group(backend="nccl")
        N_gpu = torch.distributed.get_world_size()
        print("got gpus")
    else:
        N_gpu = 1

    print(N_gpu)
    exit()
    '''


    from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator

    batch_size = 1

    eii = CXRImageIterator(batch_size, 0, 1, train_ids, train_labels, args.image_dir, image_format=args.image_format, encoding=args.label_encoding)
    pipe = CXRImagePipeline(batch_size=batch_size, num_threads=2, device_id = 0, num_gpus=4,
                                  external_data = eii)
    pii = PyTorchIterator(pipe, size=eii.size, last_batch_padded=True, fill_last_batch=False)

    times = []
    for e in range(1):
        for i, data in enumerate(pii):
            if i == 100: break 
            if i != 0:
                stop = time()
                times.append(stop-start)
            print("epoch: {}, iter {}, real batch size: {}, data shape: {}".format(e, i, len(data[0]["data"]), data[0]['data'].shape))
            img = data[0]['data'].squeeze().cpu().numpy()
            print(np.max(img))
            print(np.min(img))
            print(img.shape)
            cv2.imwrite("out.png", img*255)
            exit()
            start = time()
        pii.reset()
    print(np.mean(times))
