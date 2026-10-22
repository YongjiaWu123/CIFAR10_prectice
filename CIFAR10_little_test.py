import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import CIFAR10_model
from CIFAR10_model import *



# 创建网络模型
net = Net()

print(net)
print(net(torch.randn(2,3,32,32)).shape)






