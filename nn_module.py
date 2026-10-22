import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        output = x + 1
        return output

net = Net()
x = torch.tensor(1.0)
y = net(x)
print(y)