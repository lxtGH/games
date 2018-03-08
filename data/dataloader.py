# -*- coding:utf-8 -*-
"""
Author: XiangtaiLi
"""
import torch
import torch.utils.data as data
from torchvision import transforms, utils
import pandas as pd
import os, math, random
from skimage import io
from os.path import *
import numpy as np
from .utils import mask_dic

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
            label = self.train_labels[item]
            sample = {"image":img,"label":label}

            if self.transform:
                sample = self.transform(sample)

            sample["image"] = torch.FloatTensor(sample["image"])

            up_label, down_label = label
            mask = np.zeros(8)
            mask[mask_dic[up_label]] = 1

            y = np.zeros(10)
            y[down_label.index("y")] = 1

            return sample, torch.FloatTensor(mask), torch.FloatTensor(y)

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

    # unused
    def make_mask(self,x):
        """
        :param x: images along with their up-labels (b,c,h,w)
        :return: mask (8,b)
        """
        img, y = x
        b,c,h,w = img.shape()
        mask = np.zeros((8,b))
        for i in y:
            mask[mask_dic[y],i] = 1
        return mask




"""
if __name__ == '__main__':
    path = "/home/lxt/data/alibaba"
    Ali_dataset = FashionAttributes(root=path)
    for i in range(len(Ali_dataset)):
        sample = Ali_dataset[i]
        if i > 10:
            break
        print(sample["image"].shape)

"""






