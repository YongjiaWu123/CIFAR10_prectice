import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

dataset = datasets.CIFAR10(r"D:\AI\dataset",
                           train=True,
                           download=True,
                           transform = torchvision.transforms.ToTensor())

class CIFAR10Sequence(nn.Module):
    def __init__(self):
        super(CIFAR10Sequence, self).__init__()
        # 输入频道数为3，输出频道数为32，卷积核为5x5
        # self.conv1 = nn.Conv2d(3, 32, 5,padding=2)
        # self.maxpool1 = nn.MaxPool2d(2,2)
        # self.conv2 = nn.Conv2d(32, 32, 5,padding=2)
        # self.maxpool2 = nn.MaxPool2d(2,2)
        # self.conv3 = nn.Conv2d(32, 64, 5,padding=2)
        # self.maxpool3 = nn.MaxPool2d(2,2)
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(1024, 64)
        # self.linear2 = nn.Linear(64, 10)

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model(x)
        return x

net = CIFAR10Sequence()

# dataloader = torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=True)
# input = torch.randn(1,3,32,32)
input = torch.ones(64, 3, 32, 32)

output = net(input)
print(output.size())
# print(output)
print(net)

writer = SummaryWriter(log_dir='logs')
writer.add_graph(net,input)
writer.close()