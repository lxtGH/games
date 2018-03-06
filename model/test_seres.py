import torch
from torch import nn
from collections import OrderedDict
from .se_resnet import se_resnet50

num_class = 20
model = se_resnet50(num_classes=1000)

data = torch.load("weight-99.pkl")
state_dict =torch.load("weight-99.pkl")["weight"]


new_state_dict = OrderedDict()

for k, v in state_dict.items():
    print(k,v.size())
    name = k[7:]
    new_state_dict [name] = v

for k,v in new_state_dict.items():
    print(k,v.size())

model.load_state_dict(state_dict,strict=False)

num_ftr = model.fc.in_features
model.fc = nn.Linear(num_ftr,num_class)
print(model)