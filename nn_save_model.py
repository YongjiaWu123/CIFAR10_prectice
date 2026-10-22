import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方法1 保存模型结构+模型参数
torch.save(vgg16, 'vgg16_method1.pth')

# 保存方式2 保存模型参数;官方推荐保存方式
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')

# 陷阱
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
    def forward(self, x):
        x = self.conv1(x)
        return x
