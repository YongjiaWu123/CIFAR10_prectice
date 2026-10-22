import torch
import torch.nn as nn
from jinja2.compiler import F

input = torch.randn([4,1,3,3])
print(input)
torch.reshape(input,(-1,1,2,2))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu1 = nn.ReLU(False)
    #     inplace = True,会改变输入ReLU的值，比如input = -1 ，那么ReLU操作之后，input = 0
    #     inplace = False, 不会改变input的值，input仍然为-1，会返回output的值 0

    def forward(self, x):
        x = self.relu1(x)
        return x

net = Net()
output = net(input)
print(output)