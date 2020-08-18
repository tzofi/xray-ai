from __future__ import print_function

import cv2
import numpy as np
import torch
import h5pickle as h5py
from torchvision import datasets
from torch.utils.data import Dataset

from .cvtransforms import * 

class ImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, root, transform=None, target_transform=None,
                 two_crop=False, jigsaw_transform=None):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop
        self.jigsaw_transform = jigsaw_transform
        self.use_jigsaw = (jigsaw_transform is not None)
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.imgs[index]
        image = self.loader(path)

        # # image
        if self.transform is not None:
            img = self.transform(image)
            if self.two_crop:
                img2 = self.transform(image)
                img = torch.cat([img, img2], dim=0)
        else:
            img = image

        # # jigsaw
        if self.use_jigsaw:
            jigsaw_image = self.jigsaw_transform(image)

        if self.use_jigsaw:
            return img, index, jigsaw_image
        else:
            return img, index


class CXRDataset(Dataset):
    '''
    Base Chest X-Ray Dataset
    Given H5 path and folds, returns images, labels, and DICOM IDs
    '''

    def __init__(self, h5_path, folds, do_transform=False, pathology=False, test=False):
        self.h5_path = h5_path
        self.folds = folds
        self.test = test

        self.f = h5py.File(self.h5_path, 'r')
        self.fold_names = []
        for fold in self.folds:
            for i in range(self.f[str(fold)].shape[0]):
                self.fold_names.append((i,str(fold)))

        self.pathology = pathology
        self.do_transform = do_transform
        self.transform = Compose([
            RandomAffine(degrees=(-20,20), translate=(0.1, 0.1))
        ])

    def __len__(self):
        return len(self.fold_names)

    def __getitem__(self, idx):
        i, fold = self.fold_names[idx]
        image = self.f["{0}".format(fold)][i,...]
        label = self.f["{0}_label".format(fold)][i,...]
        img_id = self.f["{0}_dicom".format(fold)][i,0] 

        if self.pathology:
            finding = True if 1.0 in list(label[:8]) + list(label[9:-1]) else False
            if finding:
                label[8] = 0.0
            if self.test:
                label = torch.FloatTensor([l if l in [0,1] else -1 for l in label])
            else:
                label = torch.FloatTensor([l if l in [0,1] else np.random.uniform(0.8,1,1)[0] if l == -1 else 0 for l in label])
            #label = torch.FloatTensor([l if l in [0,1] else 1 if l == -1 else 0 for l in label])

        if self.do_transform:
            image = np.expand_dims(image.squeeze(), 2)
            image = np.expand_dims(self.transform(image),0)

        #image = np.concatenate([image,image,image], axis=0)
        
        #return image, label, img_id
        return image, label


class CXRContrastDataset(CXRDataset):
    '''
    SimCLR Chest X-Ray Dataset
    Inherits from CXRDataset and adds transforms
    Returns image-aug1, image-aug2, label, DICOM ID
    '''

    def __init__(self, h5_path, folds):
        super().__init__(h5_path, folds)

        self.transform = Compose([
            RandomResizedCrop(512),
            RandomHorizontalFlip()
        ])

    def __getitem__(self, idx):
        i, fold = self.fold_names[idx]
        image = self.f["{0}".format(fold)][i,...]
        label = self.f["{0}_label".format(fold)][i,...]
        img_id = self.f["{0}_dicom".format(fold)][i,0] 

        image = np.expand_dims(image.squeeze(), 2)
        img_1 = np.expand_dims(self.transforms(image),0)
        img_2 = np.expand_dims(self.transforms(image),0)
        
        #img_1 = np.concatenate([img_1,img_1,img_1], axis=0)
        #img_2 = np.concatenate([img_2,img_2,img_2], axis=0)
        img = np.concatenate([img_1, img_2], axis=0)

        return img, label, image, img_id
        #return img_1, img_2, label, img_id

    def transforms(self, img):
        img = self.transform(img)
        if np.random.uniform() < 0.5:
            ksize = int(np.round(img.shape[1] * 0.1))
            sigma = 0.1
            if np.random.randint(0,2) == 1:
                sigma = 2.0
            #img = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=sigma)
        return img

class CXRContrastClassDataset(CXRDataset):
    '''
    SimCLR Class Chest X-Ray Dataset
    Inherits from CXRDataset and adds transforms
    Returns set1 and set2 where each are length 4, with images ordered by edema severities (0,1,2,3)
    Set do_transform to True to augment the images
    '''

    def __init__(self, h5_path, folds, do_transform=False, classes=4):
        self.folds = folds
        self.f = h5py.File(h5_path, 'r')
        self.fold_names = []
        self.classes = [[] for i in range(classes)]

        for fold in folds:
            for i in range(self.f[str(fold)].shape[0]):
                self.fold_names.append((i,str(fold)))
                label = self.f["{0}_label".format(fold)][i,...]
                self.classes[np.argmax(label)].append((i, str(fold)))

        self.do_transform = do_transform
        self.transform = Compose([
            RandomResizedCrop(512),
            RandomHorizontalFlip()
        ])

    def __getitem__(self, idx):
        i, fold = self.fold_names[idx]

        imgs1 = []
        imgs2 = []
        for c in range(len(self.classes)):
            samples = np.random.randint(0, len(self.classes[c]), size=2)
            i, fold = self.classes[c][samples[0]]
            img1 = self.f["{0}".format(fold)][i,...]
            i, fold = self.classes[c][samples[1]]
            img2 = self.f["{0}".format(fold)][i,...]

            if self.do_transform:
                img1 = np.expand_dims(img1.squeeze(), 2)
                img1 = np.expand_dims(self.transforms(img1),0)
                img2 = np.expand_dims(img2.squeeze(), 2)
                img2 = np.expand_dims(self.transforms(img2),0)

            imgs1.append(img1)
            imgs2.append(img2)

        #imgs = np.concatenate([np.array(imgs1), np.array(imgs2)], axis=0)

        #return imgs.squeeze()
        return np.array(imgs1), np.array(imgs2)
