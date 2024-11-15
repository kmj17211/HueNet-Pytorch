import os
import random
import pathlib
import numpy as np
from os.path import join
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset

from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CT_Dataset(Dataset):
    def __init__(self, transform, args, hist = False):
        super().__init__()
        
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.task = args.task
        self.min_val = args.min_val
        self.max_val = args.max_val
        self.bin_num = args.bin_num
        self.hist = hist

        if transform:
            self.transform = transform
        else:
            self.transform = None

        data_root = pathlib.Path(self.data_dir)
        all_image_path = list(data_root.glob('*'))
        all_image_path = [str(path) for path in all_image_path]

        random.seed(1)
        random.shuffle(all_image_path)

        # split to train and validation
        train_ratio = 0.9
        train_size = round(len(all_image_path) * train_ratio)
        self.train_paths = all_image_path[0:train_size]
        self.val_paths = all_image_path[train_size:]

    def load_hist(self, img):
        hist1, _ = torch.histogram(img[0], range = [self.min_val, self.max_val], bins = self.bin_num)
        hist2, _ = torch.histogram(img[1], range = [self.min_val, self.max_val], bins = self.bin_num)
        hist3, _ = torch.histogram(img[2], range = [self.min_val, self.max_val], bins = self.bin_num)

        return hist1.type(torch.int32).to(device), hist2.type(torch.int32).to(device), hist3.type(torch.int32).to(device)

class CT_Dataset_Train(CT_Dataset):
    def __getitem__(self, index):
        image_path = self.train_paths[index]
        image = Image.open(image_path).convert('YCbCr')

        if self.transform:
            image = self.transform(image)

        image = image * 2 - 1

        if self.hist:
            return image, self.load_hist(image)
        else:
            return image
    
    def __len__(self):
        return len(self.train_paths)
    
class CT_Dataset_Test(CT_Dataset):
    def __getitem__(self, index):
        image_path = self.val_paths[index]
        image = Image.open(image_path).convert('YCbCr')

        if self.transform:
            image = self.transform(image)

        image = image * 2 - 1

        if self.hist:
            return image, self.load_hist(image)
        else:
            return image
    
    def __len__(self):
        return len(self.val_paths)
    
class SAMPLE_Dataset(Dataset):
    def __init__(self, transform, args, real, aug = False, hist = False, train = True):
        super(SAMPLE_Dataset).__init__()

        path = args.data_dir
        self.ep = args.ep
        self.min_val = args.min_val
        self.max_val = args.max_val
        self.bin_num = args.bin_num
        self.hist = hist
        self.train = train
        self.aug = aug

        if transform:
            self.transform = transform
        else:
            self.transform = False

        if real:
            self.dir_r = 'real'
        else:
            self.dir_r = 'synth'

        self.label = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']

        if train:
            self.dir_t = 'train'
            self.label = self.label[:7]
        else:
            self.dir_t = 'test'

        # Data Path
        self.data_path = []
        self.data_label = []
        self.file_name = []
        for label in self.label:
            path2data = join(path, self.dir_r, self.dir_t, label)
            for file_name in os.listdir(path2data):
                data_path = join(path2data, file_name)

                self.data_path.append(data_path)
                self.data_label.append(label)
                self.file_name.append(file_name)
    
    def hist_aug(self, img):
        if random.randint(0, 100) > 50:
            clip = random.randint(1, 20)
            img[img > clip] = clip

        return img
        
    def __len__(self):
        return len(self.data_path)
        
    def load_hist(self, img):
        hist1, _ = torch.histogram(img[0], range = [self.min_val, self.max_val], bins = self.bin_num)

        return hist1.type(torch.int32).to(device)


class SAMPLE_Real_Dataset(SAMPLE_Dataset):
    def __getitem__(self, index):
        
        # Label
        label = self.data_label[index]

        # File Name
        file_name = self.file_name[index]
        
        # Data Load
        img = abs(loadmat(self.data_path[index])['complex_img'])
        if self.aug:
            img = self.hist_aug(img)
        img = np.log10(img + self.ep) - np.log10(self.ep)
        img = img / np.max(img)

        img = img * 2 - 1

        img = self.transform(img).type(torch.float32)
        if self.hist:
            if self.train:
                return img, self.load_hist(img)
            else:
                return img, self.load_hist(img), label, file_name
        else:
            if self.train:
                return img
            else:
                return img.type(torch.float32), label, file_name
    
class SAMPLE_Synth_Dataset(SAMPLE_Dataset):
    def __getitem__(self, index):
        
        # Label
        label = self.data_label[index]

        # File Name
        file_name = self.file_name[index]
        file_name = file_name.replace('synth', 'refine')
        
        # Data Load
        img = abs(loadmat(self.data_path[index])['complex_img'])
        img = np.log10(img + self.ep) - np.log10(self.ep)
        img = img / np.max(img)

        img = img * 2 - 1

        img = self.transform(img).type(torch.float32)
        if self.hist:
            if self.train:
                return img, self.load_hist(img)
            else:
                return img, self.load_hist(img), label, file_name
        else:
            if self.train:
                return img
            else:
                return img.type(torch.float32), label, file_name
