from data.utils import *
from data.dataloader import FashionAttributes
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = "/home/lxt/data/ali"
    Ali_dataset = FashionAttributes(root=path,
                                    transform=transforms.Compose(
                                        [
                                            Rescale(256),
                                            RandomCrop(224),
                                            ToTensor()
                                        ]))
    dataloader = DataLoader(Ali_dataset,batch_size=128, shuffle=True, num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch, sample_batched[0]['image'].size(),
               # sample_batched[1],
              #sample_batched[0]['label'],sample_batched[2].size())

        y = sample_batched[2]

        print(y[64:])

        #mask = sample_batched[1].size()
        #print(mask)
        # observe 4th batch and stop.
        if i_batch == 3:
            break

    mask = torch.IntTensor(np.ones((4,5,6),dtype=int))
