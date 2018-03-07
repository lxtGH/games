import torch
from torch import nn
from collections import OrderedDict
from model.se_resnet import se_resnet50
from torch.autograd import Variable


class Equal(nn.Module):
    def __init__(self,x):
        self.x = x
    def forward(self, x):
        return x

num_class = 20
model = se_resnet50(num_classes=1000)

data = torch.load("/home/lxt/Github/games/model/pretrained/weight-99.pkl")
state_dict =torch.load("/home/lxt/Github/games/model/pretrained/weight-99.pkl")["weight"]


new_state_dict = OrderedDict()

for k, v in state_dict.items():
    print(k,v.size())
    name = k[7:]
    new_state_dict [name] = v
    if name =="fc":
        break

for k,v in new_state_dict.items():
    print(k,v.size())

model.load_state_dict(state_dict,strict=False)


print(model)

x = torch.randn(20,3,224,224)

y = model(Variable(x))

print(y.size())