import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

dataset = datasets.CIFAR10(r"D:\AI\dataset",
                           train=False,
                           download=True,
                           transform = torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True)
class CIFAR10Sequence(nn.Module):
    def __init__(self):
        super(CIFAR10Sequence, self).__init__()

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
        x = self.model(x)
        return x

net = CIFAR10Sequence()

loss = nn.CrossEntropyLoss()

for data in dataloader:
    inputs, labels = data
    outputs = net(inputs)
    # print(outputs)
    # print(labels)
    loss_result = loss(outputs, labels)
    print(loss_result)
    loss_result.backward()


