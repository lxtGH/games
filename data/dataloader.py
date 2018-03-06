# -*- coding:utf-8 -*-
"""
Author: XiangtaiLi
"""

import torch
import torch.utils.data as data
from torchvision import transforms, utils
from .utils import Rescale,RandomCrop,ToTensor
import pandas as pd
import os, math, random
from skimage import io
from os.path import *
import numpy as np



class FashionAttributes(data.Dataset):
    """
    Fashion Attributes data object set for load the data and label
    """
    def __init__(self,root, mode = "train",transform = None,dataset_name="FashionAttributes"):
        self.root = root
        self.transform = transform
        self.name = dataset_name
        self.mode = mode
        if self.mode == "train":
            self.train_path = os.path.join(self.root,"base")
            self.train_anno = os.path.join(self.train_path,"Annotations","label.csv")
            self.train_data = pd.read_csv(str(self.train_anno), header=None)
            self.train_imgs = self.train_data[0]
            self.train_labels = [ (i,j) for (i,j) in zip(self.train_data[1],self.train_data[2])]
        elif self.mode == "test":
            self.test_path = os.path.join(self.root,"rank")
            self.test_data = pd.read_csv(os.path.join(self.test_path,"Test","question.csv"),header=None)
            self.test_imgs = self.test_data[0]

    def __getitem__(self, item):
        """
        :param item:
        :return:
        """
        if self.mode == "train":
            img = io.imread(os.path.join(self.train_path,self.train_imgs[item]))
            if self.transform:
                img = self.transform(img)
            label = self.train_labels[item]
            return {"image":img,"label":label}
        if self.mode =="test":
            img = io.imread(os.path.join(self.test_path,self.test_imgs[item]))
            if self.transform:
                img = self.transform(img)
            return {"image":img}

    def __len__(self):
        if self.mode == "train":
            return len(self.train_data)
        if self.mode =="test":
            return len(self.test_data)


if __name__ == '__main__':
    path = "/home/lxt/data/alibaba"
    Ali_dataset = FashionAttributes(root=path)

    for i in range(len(Ali_dataset)):
        sample = Ali_dataset[i]
        if i > 10:
            break
        print(sample["image"].shape)





