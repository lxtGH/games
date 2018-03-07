from data.utils import *
from data.dataloader import FashionAttributes
from model.baseline_model import baseline_model
from torch.utils.data import Dataset, DataLoader
from model.baseline_model import Loss_func
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.autograd import Variable


def make_mask(x, mask):
    """
    :param x: (b,f_size)
    :param mask: (b,8) 8代表8个不同的大的类别
    :return: feature mask  (8,b,f)
    """
    b = x.size()[0]
    f = 2048 # feature size
    f_mask = np.zeros((8, b, f), dtype=int)

    for i in range(8):
        index = np.nonzero(mask[:, i] == 1)
        index = np.array(index)
        f_mask[i, index, i] = 1
    #print(f_mask)
    return f_mask


if __name__ == '__main__':
    path = "/home/lxt/data/alibaba"
    Ali_dataset = FashionAttributes(root=path,
                                    transform=transforms.Compose(
                                        [
                                            Rescale(256),
                                            RandomCrop(224),
                                            ToTensor()
                                        ]))
    dataloader = DataLoader(Ali_dataset, batch_size=20, shuffle=True, num_workers=4)

    base_model = baseline_model().cuda()


    print(base_model.modules())
    print(base_model)


    print()

    for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch, sample_batched[0]['image'].size(),
              #sample_batched[1],
              #sample_batched[0]['label'], sample_batched[2].size())
        # observe 4th batch and stop.


        if i_batch == 3:
            img = sample_batched[0]['image']
            mask = sample_batched[1]
            y = sample_batched[2]
            img = torch.FloatTensor(img.numpy())
            mask = torch.FloatTensor(mask.numpy())
            y = torch.FloatTensor(mask.numpy())

            img = Variable(img).cuda()
            mask = Variable(mask).cuda()
            y = Variable(y).cuda()
            #input = (img, mask)
            out = base_model.forward(img)
            #print(out[0].size())
            #print(out[1].size())
            #print(out[5].size())
            #print(y.size())
            #print(mask.size())
            loss = Loss_func(out,y,mask)
            print(loss())
            break

    mask = torch.IntTensor(np.ones((4, 5, 6), dtype=int))
