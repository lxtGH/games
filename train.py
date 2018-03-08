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
from model.baseline_model import Loss_func
from data.utils import *
from data.dataloader import FashionAttributes

path = "/home/lxt/data/ali"
Ali_dataset = FashionAttributes(root=path,
                                transform=transforms.Compose(
                                    [
                                        Rescale(256),
                                        RandomCrop(224),
                                        ToTensor()
                                    ]))
softmax = nn.Softmax()
dataloader = DataLoader(Ali_dataset, batch_size=64, shuffle=True, num_workers=4)
net = baseline_model()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
use_cuda = True

def train(epoch):
    print("epoch: %d" %epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, sample_batched in enumerate(dataloader):
        optimizer.zero_grad()
        img, mask,  y = sample_batched[0]['image'], sample_batched[1], sample_batched[3]

        img = Variable(img)
        mask = Variable(mask)
        y = Variable(y)
        if use_cuda:
            img.cuda()
            mask.cuda()
            net.cuda()
            y.cuda()
        input = (img, mask)
        out = net.forward(input)

        loss = Loss_func(out, y, mask,softmax)
        loss_array, output= loss()
        loss_total = sum(loss_array)
        train_loss += loss_total.data[0]

        loss_total.backward()

        if batch_idx % 100 == 0:
            print("Training: Epoch: %d, batch: %d, Loss: %.3f , Acc: %.3f, Correct/Total: (%d/%d)"
                  % (epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))






