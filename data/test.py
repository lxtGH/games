import pandas as pd
import os

path = "/home/lxt/data/alibaba/base"

ano_path = os.path.join(path,"Annotations","label.csv")

print(ano_path)

data = pd.read_csv(str(ano_path),header=None)

print(data[0])
print(data[2])

imgList = data[0]
label = data[2]
label = list(label)
print(type(label))
print(label)
top = data[1]
print(list(top))

test = [(i, j) for i ,j in zip(data[1], data[2])]
print(test[0])