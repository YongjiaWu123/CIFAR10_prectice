import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import CIFAR10_model
from CIFAR10_model import *

train_data = torchvision.datasets.CIFAR10(root=r"D:\AI\dataset",
                                          train=True,
                                          transform = torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root=r"D:\AI\dataset",
                                         train=False,
                                         transform = torchvision.transforms.ToTensor(),
                                         download=True)
# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=True)

# 创建网络模型
net = Net()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 学习速率
loss_rate = 0.01
# 优化器
optimizer = torch.optim.SGD(net.parameters(), loss_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮次数
epoch = 10

# 添加tensorboard
writer = SummaryWriter(log_dir='logs')

for i in range(epoch):
    print("-----------第{}轮训练开始----------".format(i+1))

    #训练步骤开始
    # 不一定非要将网络设置为train才能开始训练
    # 只有一些特定的网络层必须要进入训练模式才可以训练，比如dropout层
    # net.train()
    for data in train_loader:
        imgs, targets = data
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数{}, loss:{} ".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    # 不一定要将网络设置为eval才可以测试
    # net.eval()
        # 正确率
    total_accuracy = 0
    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(dim = 1) ==targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    total_test_step += 1
    writer.add_scalar("test_loss", total_test_loss, total_train_step)
    writer.add_scalar("test_accuracy", total_accuracy, total_train_step)

    torch.save(net,"CIFAR10_{}_net.pth".format(i))
    print("模型已保存")
writer.close()






