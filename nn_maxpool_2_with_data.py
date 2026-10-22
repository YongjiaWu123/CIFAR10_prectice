import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter


dataset=torchvision.datasets.CIFAR10(root=r'D:\AI\dataset', train=False, download=True,transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.maxpool1(x)
        return x

writer = SummaryWriter(log_dir="logs")
net = Net()
step = 0
for data in dataloader:
    images, labels = data
    writer.add_images("maxpool_input", images, step)
    outputs = net(images)
    writer.add_images("maxpool_output",outputs,step)
    step += 1
    if step > 10:
        break

writer.close()