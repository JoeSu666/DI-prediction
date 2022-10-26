import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import random
import numpy as np
import glob
import shutil
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


class DIPn(Dataset):
    def __init__(self, train='train', transform=None, split=42):

        self.img_dir = './data/DI_Pneck'

        self.split = split
        
        trainnamelist, valnamelist, testnamelist = self.balance_split()

        if train == 'train':
            self.img_names = trainnamelist
        elif train == 'test':
            self.img_names = testnamelist
        elif train == 'val':
            self.img_names = valnamelist
            
        self.transform = transform
    
    def balance_split(self):

        posnamelist = glob.glob(os.path.join(self.img_dir, 'positive', '*.jpg'))
        negnamelist = glob.glob(os.path.join(self.img_dir, 'negative', '*.jpg'))
        postrainlist, postestlist = train_test_split(posnamelist, test_size=26, random_state=42)
        negtrainlist, negtestlist = train_test_split(negnamelist, test_size=26, random_state=42)
        postrainlist, posvallist = train_test_split(postrainlist, test_size=17, random_state=self.split)
        negtrainlist, negvallist = train_test_split(negtrainlist, train_size=131, test_size=17, random_state=self.split)

        trainnamelist = []
        for p, n in zip(postrainlist, negtrainlist):
            trainnamelist.append(p)
            trainnamelist.append(n)
        testnamelist = postestlist + negtestlist
        valnamelist = posvallist + negvallist
        
        return trainnamelist, valnamelist, testnamelist

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        image = Image.open(img_path)

        label = img_path.split('/')[-2]
        if label == 'positive':
            label = 1
        else:
            label = 0
        if self.transform:
            image = self.transform(image)

        return image, label



class DIfna(Dataset):
    def __init__(self, train='train', transform=None, split=42):

        self.img_dir = './data/DI_Pneck'

        self.split = split
        
        trainnamelist, valnamelist, testnamelist = self.balance_split()

        if train == 'train':
            self.img_names = trainnamelist
        elif train == 'test':
            self.img_names = testnamelist
        elif train == 'val':
            self.img_names = valnamelist
            
        self.transform = transform
    
    def balance_split(self):
        posnamelist = glob.glob(os.path.join(self.img_dir, 'positive', '*.jpg'))
        negnamelist = glob.glob(os.path.join(self.img_dir, 'negative', '*.jpg'))
        postrainlist, postestlist = train_test_split(posnamelist, test_size=26, random_state=42)
        negtrainlist, negtestlist = train_test_split(negnamelist, test_size=26, random_state=42)
        postrainlist, posvallist = train_test_split(postrainlist, test_size=17, random_state=self.split)
        negtrainlist, negvallist = train_test_split(negtrainlist, train_size=131, test_size=17, random_state=self.split)

        trainnamelist = []
        for p, n in zip(postrainlist, negtrainlist):
            trainnamelist.append(p)
            trainnamelist.append(n)
        testnamelist = postestlist + negtestlist
        valnamelist = posvallist + negvallist
        
        return trainnamelist, valnamelist, testnamelist

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        img_path = os.path.join('data/DI_Fnecka', img_path.split('/')[-2], img_path.split('/')[-1])
        image = Image.open(img_path)

        label = img_path.split('/')[-2]
        if label == 'positive':
            label = 1
        else:
            label = 0
        if self.transform:
            image = self.transform(image)

        return image, label

