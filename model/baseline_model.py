import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from .se_resnet import se_resnet50


class baseline_model(nn.Module):

    def __init__(self,class_number=[8,5,5,5,10,6,6,9],mode="train",path="./model/pretrained/weight-99.pkl"):
        super(baseline_model, self).__init__() 
        self.se_model = se_resnet50(num_classes=1000)
        self.mode = mode
        self.state_dict = torch.load(path)["weight"]
        self.new_state_dict = OrderedDict()
        self.fc_list = []

        for i in class_number:
            self.fc_list.append(nn.Linear(2048,i,bias=False))

        self.fc_net_1 = self.fc_list[0]
        self.fc_net_2 = self.fc_list[1]
        self.fc_net_3 = self.fc_list[2]
        self.fc_net_4 = self.fc_list[3]
        self.fc_net_5 = self.fc_list[4]
        self.fc_net_6 = self.fc_list[5]
        self.fc_net_7 = self.fc_list[6]
        self.fc_net_8 = self.fc_list[7]

        self.model_init()

        for k, v in self.state_dict.items():
            name = k[7:]
            self.new_state_dict[name] = v

        self.se_model.load_state_dict(self.state_dict, strict=False)


    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias:
                    init.constant(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)

            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-3)



    def make_mask(self,x, mask):
        """
        :param x: (b,f_size)
        :param mask: (b,8) 8代表8个不同的大的类别
        :return: feature mask  (8,b,f)
        """
        b, f = x.size()
        f_mask = torch.FloatTensor(np.zeros((8,b,f),dtype=int))

        print(b)
        print(f_mask.size())
        print(type(mask))
        for i in range(8):
           index = list(torch.nonzero(mask[:,i] == 1))
           index = np.array(index)
           print(index)
           f_mask[i,index,i]= 1
        print(f_mask)
        return Variable(f_mask).cuda()


    def forward(self,x):
        x = self.se_model(x)
        output = []
        for i in range(8):
            out = self.fc_list[i](x)
            output.append(out)
        return output


class Loss_func(object):
    def __init__(self, out, gt,mask,softmax,loss_f=nn.MSELoss()):
        self.out = out
        self.mask = mask
        self.gt = gt
        self.loss_f = loss_f
        self.total = []
        self.softmax = softmax
        self.output = []

    def __call__(self,):
        batch = self.out[0].size()[0]
        output = Variable(torch.FloatTensor(np.zeros((batch,10)))).cuda()
        for i in range(len(self.out)):

            dis = self.out[i].size()[1]
            mask = self.mask[:,i].contiguous().squeeze()
            mask = torch.nonzero(mask).squeeze() #get the index
            #print("mask index",mask.size())
            #print("out: ",self.out[i].size())
            #mask = mask.view(batch,-1).expand([-1,dis]) # get the mask
            #print(mask)
            res = torch.index_select(self.out[i][:,:dis],0,mask)
            print("resout: ",res.size())
            res = self.softmax(res)
            #print(res)

            y = torch.index_select(self.gt[:,:dis],0,mask)

            #exit()
            self.total.append(self.loss_f(res,y))
            self.output.append(res)

        return self.total, self.output

