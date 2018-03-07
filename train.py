# -*- coding:utf-8 -*-
# Author: Xiangtai Li



from model.baseline_model import baseline_model
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import argparse

from data.utils import *
from data.dataloader import FashionAttributes

def train():
    path = "/home/lxt/data/alibaba"
    Ali_dataset = FashionAttributes(root=path,
                                    transform=transforms.Compose(
                                        [
                                            Rescale(256),
                                            RandomCrop(224),
                                            ToTensor()
                                        ]))
    dataloader = DataLoader(Ali_dataset, batch_size=64, shuffle=True, num_workers=4)
    net = baseline_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    use_cuda = True

    for i_batch, sample_batched in enumerate(dataloader):
        img, mask, y= sample_batched[0]['image'], sample_batched[1],sample_batched[3]

        img = torch.FloatTensor(img.numpy())
        mask = torch.IntTensor(mask.numpy())
        img = Variable(img)
        mask = Variable(mask)


        if use_cuda:
            img.cuda()
            net.cuda()

        input = (img, mask)

        out = net.forward(input)

        y = Variable(y)

        loss = criterion(out, y)
        loss.backward()




