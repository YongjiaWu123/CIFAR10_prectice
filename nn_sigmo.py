import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root=r'D:\AI\dataset', train=True, download=True,transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.sigmoid(x)
        return x
writer = SummaryWriter(log_dir="logs")
step = 0
for data in dataloader:
    images, labels = data
    writer.add_images("sigmoid_input", images, step)
    outputs = Net()(images)
    writer.add_images("sigmoid_output", outputs, step)
    if step > 10:
        break
    step += 1

writer.close()

