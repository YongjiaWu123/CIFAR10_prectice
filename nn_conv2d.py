import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10(root=r'D:\AI\dataset',
                                     train=False,
                                     download=True,
                                     transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3,stride=1, padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x
net = Net()
print(net)

writer = SummaryWriter(log_dir="logs")
step = 0
for data in dataloader:
    imgs,targets = data
    outputs = net(imgs)
    # print(targets.shape)
    print(imgs.shape)
    print(outputs.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("inputs",imgs,step)
    # torch.Size([64, 6, 30, 30])
    outputs = torch.reshape(outputs,(-1,3,30,30))           #将通道切开，让batchsize变多
    writer.add_images("outputs",outputs,step)
    # 以上代码报错原因，彩色图像只能显示3个channels，但是输出是有6个channels
    step+=1
    if step >10:
        break