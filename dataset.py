import os
import cv2
import h5py
import torch
import itertools
import numpy as np
import pandas as pd
import torchvision.transforms.functional as trnF

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import utils.opencv_functional as cv2f
import utils.cvtransforms as cvtransforms 

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


class CXRDataset(Dataset):
    '''
    Base Chest X-Ray Dataset
    Given H5 path and folds, returns images, labels, and DICOM IDs
    '''

    def __init__(self, h5_path, folds, do_transform=False):
        self.folds = folds
        self.f = h5py.File(h5_path, 'r')
        self.fold_names = []

        for fold in folds:
            for i in range(self.f[str(fold)].shape[0]):
                self.fold_names.append((i,str(fold)))

        self.do_transform = do_transform
        self.transform = cvtransforms.Compose([
            cvtransforms.RandomAffine(degrees=(-20,20), translate=(0.1, 0.1))
        ])

    def __len__(self):
        return len(self.fold_names)

    def __getitem__(self, idx):
        i, fold = self.fold_names[idx]
        image = self.f["{0}".format(fold)][i,...]
        label = self.f["{0}_label".format(fold)][i,...]
        img_id = self.f["{0}_dicom".format(fold)][i,0] 

        if self.do_transform:
            image = np.expand_dims(image.squeeze(), 2)
            image = np.expand_dims(self.transform(image),0)
        
        return image, label, img_id

class CXRSimCLRDataset(CXRDataset):
    '''
    SimCLR Chest X-Ray Dataset
    Inherits from CXRDataset and adds transforms
    Returns image-aug1, image-aug2, label, DICOM ID
    '''

    def __init__(self, h5_path, folds):
        super().__init__(h5_path, folds)

        self.transform = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(512),
            cvtransforms.RandomHorizontalFlip()
        ])

    def __getitem__(self, idx):
        i, fold = self.fold_names[idx]
        image = self.f["{0}".format(fold)][i,...]
        label = self.f["{0}_label".format(fold)][i,...]
        img_id = self.f["{0}_dicom".format(fold)][i,0] 

        image = np.expand_dims(image.squeeze(), 2)
        img_1 = np.expand_dims(self.transforms(image),0)
        img_2 = np.expand_dims(self.transforms(image),0)
        
        return img_1, img_2, label, img_id

    def transforms(self, img):
        img = self.transform(img)
        if np.random.uniform() < 0.5:
            ksize = int(np.round(img.shape[1] * 0.1))
            sigma = 0.1
            if np.random.randint(0,2) == 1:
                sigma = 2.0
            img = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=sigma)
        return img


class CXRSSLDataset(CXRDataset):
    '''
    SSL Chest X-Ray Dataset
    Inherits from CXRDataset and adds transforms for rotation/shifting
    Returns image stack, label, DICOM ID
    '''

    def __init__(self, h5_path, folds, expanded_params):
        super().__init__(h5_path, folds)

        self.expanded_params = expanded_params
        self.pert_configs = []
        for tx, ty in itertools.product(*expanded_params):
            self.pert_configs.append((tx, ty))
        self.num_perts = len(self.pert_configs)

    def __getitem__(self, idx):
        i, fold = self.fold_names[idx]
        image = self.f["{0}".format(fold)][i,...]
        label = self.f["{0}_label".format(fold)][i,...]
        img_id = self.f["{0}_dicom".format(fold)][i,0] 

        image = image.reshape(image.shape[1], image.shape[2], 1)

        pert = self.pert_configs[idx % self.num_perts]
        if np.random.uniform() < 0.5:
            ''' random mirroring '''
            image = image[:, ::-1]
        image = cv2f.affine(np.asarray(image), 0, (pert[0], pert[1]), 1, 0,
                        interpolation=cv2.INTER_LINEAR, mode=cv2.BORDER_REFLECT_101)

        label = [self.expanded_params[i].index(pert[i]) for i in range(len(self.expanded_params))]
        label = np.vstack((label + [0], label + [1], label + [2], label + [3]))

        image = trnF.to_tensor(image.copy()).unsqueeze(0).numpy()
        image = np.concatenate((image, np.rot90(image, 1, axes=(2, 3)),
                            np.rot90(image, 2, axes=(2, 3)), np.rot90(image, 3, axes=(2, 3))), 0)

        return torch.FloatTensor(image), label, img_id

    
class CXRresizer(Dataset):
    '''
    CXRresizer takes a directory of NPY files and CSV and generates H5
    Example:
    dataset = CXRresizer("labels.csv", "/data/npy", size=(2048,2048))
    dataset.transfer("2048_dataset.h5")
    '''

    def __init__(self, labels_csv, root_dir, pad='constant', size=(512,512)):

        self.root_dir = root_dir
        self.pad = pad
        self.size = size
        df = pd.read_csv(labels_csv)
        self.img_ids = np.array(df['dicom_id']).astype(str)
        self.labels = np.array(df['edeme_severity']).astype(int)
        self.folds = np.array(df['fold']).astype(str)

        indices = {}
        for fold in self.folds:
            if fold not in indices:
                indices[fold] = [0]
            else:
                indices[fold].append(indices[fold][-1]+1)

        self.indices = indices
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        fold = self.folds[idx]
        label = self.labels[idx]
        img_path = os.path.join(self.root_dir,
                                img_id+'.npy')
        image = np.load(img_path).squeeze()
        a = image.shape[0]
        b = image.shape[1]
        if a > b:
            pads = ((0,0),((a-b)//2,(a-b)//2))
        elif b > a:
            pads = (((b-a)//2,(b-a)//2),(0,0))
        else:
            pads = ((0,0),(0,0))
        image = np.pad(image,pads,self.pad)
        image = cv2.resize(image, dsize=self.size, interpolation=cv2.INTER_AREA)
        label = np.array(convert_to_onehot(label)).reshape(4,).astype(np.float32)
        sample = [image, label, fold, img_id]

        return sample
    
    def __len__(self):
        return len(self.img_ids)

    def transfer(self, path):
        f = h5py.File(path, mode='w')

        f.create_dataset("1", (len(self.indices["1"]), 1, self.size[0], self.size[1]), np.float32)
        f.create_dataset("2", (len(self.indices["2"]), 1, self.size[0], self.size[1]), np.float32)
        f.create_dataset("3", (len(self.indices["3"]), 1, self.size[0], self.size[1]), np.float32)
        f.create_dataset("4", (len(self.indices["4"]), 1, self.size[0], self.size[1]), np.float32)
        f.create_dataset("5", (len(self.indices["5"]), 1, self.size[0], self.size[1]), np.float32)
        f.create_dataset("TEST", (len(self.indices["TEST"]), 1, self.size[0], self.size[1]), np.float32)
        f.create_dataset("1_label", (len(self.indices["1"]), 4), np.float32)
        f.create_dataset("2_label", (len(self.indices["2"]), 4), np.float32)
        f.create_dataset("3_label", (len(self.indices["3"]), 4), np.float32)
        f.create_dataset("4_label", (len(self.indices["4"]), 4), np.float32)
        f.create_dataset("5_label", (len(self.indices["5"]), 4), np.float32)
        f.create_dataset("TEST_label", (len(self.indices["TEST"]), 4), np.float32)
        f.create_dataset("1_dicom", (len(self.indices["1"]), 1), np.int32)
        f.create_dataset("2_dicom", (len(self.indices["2"]), 1), np.int32)
        f.create_dataset("3_dicom", (len(self.indices["3"]), 1), np.int32)
        f.create_dataset("4_dicom", (len(self.indices["4"]), 1), np.int32)
        f.create_dataset("5_dicom", (len(self.indices["5"]), 1), np.int32)
        f.create_dataset("TEST_dicom", (len(self.indices["TEST"]), 1), np.int32)

        for i in tqdm(range(self.__len__())):
            fold = self.folds[i]
            index = self.indices[fold][0]
            self.indices[fold].pop(0)
            data = self.__getitem__(i)
            f["{0}".format(data[2])][index,0,:,:] = data[0].astype(np.float32)
            f["{0}_label".format(data[2])][index,:] = data[1]
            f["{0}_dicom".format(data[2])][index,:] = [i]
        f.close()
        print(self.indices)

#dataset = CXRresizer("/home/pe24975/labels.csv", "/mnt/USB/npy", size=(2048,2048))
#dataset.transfer("2048_dataset.h5")
'''
from time import time
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import kornia

transform = nn.Sequential(
    kornia.augmentation.RandomAffine((-20., 20.),translate=(0.1, 0.1))
    #kornia.augmentation.RandomRotation(degrees=[-50.0,50.0])
    #kornia.color.AdjustBrightness(0.5),
    #kornia.color.AdjustGamma(gamma=2.),
    #kornia.color.AdjustContrast(0.7),
)

pixel_shift = 56 #0.25 * args.image_resize
expanded_params = ((0, -1 * pixel_shift, pixel_shift),(0, -1 * pixel_shift, pixel_shift))

device = torch.device("cuda")
dataset = CXRDataset("1024_dataset.h5",["1","2","3"])
train_loader = DataLoader(dataset, batch_size=2,
                         shuffle=False, drop_last=True, num_workers=1)

times = []
for i, batch in enumerate(train_loader):
    if i == 100: break 
    if i != 0:
        stop = time()
        times.append(stop-start)
    X,Y,idx = batch[0], batch[1], batch[2] #dataset.__getitem__(i)
    X = X.to(device)
    Y = Y.to(device)

    X1 = X.squeeze()#.cpu().numpy() * 255
    for j, im in enumerate(X1):
        im = im.cpu().numpy() * 255
        cv2.imwrite("out-{}.png".format(j), im)

    X = transform(X)

    X = X.squeeze()#.cpu().numpy() * 255
    for j, im in enumerate(X):
        im = im.cpu().numpy() * 255
        cv2.imwrite("transform-{}.png".format(j), im)
    #X = X.squeeze().cpu().numpy() * 255
    #cv2.imwrite("out-transform.png", X)
    print(X.shape)
    exit()
    start = time()
    
print(np.mean(times))
'''

'''
d = CXRDataset('/data3/cxr_data/512_dataset.h5', [1,2,3,4])
train_loader = DataLoader(d, batch_size=1,shuffle=False,drop_last=True)
for i, (xi, xj, _, ) in enumerate(train_loader):
    print(xi.shape)
    cv2.imwrite("output-1.png", np.array(xi.squeeze() * 255, dtype=np.uint8))
    exit()
'''
