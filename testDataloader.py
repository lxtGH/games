from data.utils import *
from data.dataloader import FashionAttributes
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = "/home/lxt/data/alibaba"
    Ali_dataset = FashionAttributes(root=path,
                                    transform=transforms.Compose(
                                        [
                                            Rescale(256),
                                            RandomCrop(224),
                                            ToTensor()
                                        ]))
    dataloader = DataLoader(Ali_dataset,batch_size=20, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['label'])

        # observe 4th batch and stop.
        if i_batch == 3:
            #plt.figure()
            #plt.plot()
            #plt.axis('off')
            #plt.ioff()
            #plt.show()
            break