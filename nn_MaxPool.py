import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms

dataset=torchvision.datasets.CIFAR10(root=r'D:\AI\dataset', train=False, download=True,transform=transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
input = torch.randn(1, 1, 5, 5)
input = torch.reshape(input, [-1, 1, 5, 5])
print(input.shape)
print(input)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_floor = nn.MaxPool2d(2, 2, ceil_mode=False)
        self.pool_ceil = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, x, use_ceil=False):
        if use_ceil:
            x = self.pool_ceil(x)
        else:
            x = self.pool_floor(x)
        return x
net = Net()
output1 = net(input)
print(output1.shape)
print(output1)
output2 = net(input, use_ceil=True)
print(output2.shape)
print(output2)