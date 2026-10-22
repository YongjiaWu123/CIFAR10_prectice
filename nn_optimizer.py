import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

# 数据集
dataset = datasets.CIFAR10(r"D:\AI\dataset",
                           train=False,
                           download=True,
                           transform = torchvision.transforms.ToTensor())
# 加载
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
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
for epoch in range(20):
    for data in dataloader:
        inputs, labels = data
        outputs = net(inputs)
        loss_result = loss(outputs, labels)
        # 要先清空梯度
        optimizer.zero_grad()
        loss_result.backward()
        optimizer.step()
        print(loss_result)



