import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import CIFAR10_model
from CIFAR10_model import *

torch.manual_seed(123)


full_train_data = torchvision.datasets.CIFAR10(root=r"D:\AI\dataset",
                                          train=True,
                                          transform = torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root=r"D:\AI\dataset",
                                         train=False,
                                         transform = torchvision.transforms.ToTensor(),
                                         download=True)
train_data, val_data = torch.utils.data.random_split(
    full_train_data,
    [45000, 5000],
    generator=torch.Generator().manual_seed(42)
)
# 数据集长度
train_data_size = len(train_data)
val_data_size = len(val_data)
test_data_size = len(test_data)

print("训练数据集长度为：{}".format(train_data_size))
print("验证数据集长度为：{}".format(val_data_size))
print("测试数据集长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(123)
)

val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=64, shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=64, shuffle=False
)

# print("划分结果：", len(train_data), len(val_data), len(test_data))
# raise SystemExit(0)  # 临时正常退出，先检查数据划分

# 创建网络模型
if not torch.cuda.is_available():
    raise RuntimeError("当前 PyTorch 环境无法使用 CUDA")
device = torch.device("cuda")
print("训练设备：", device, torch.cuda.get_device_name(0))
net = Net().to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 学习速率
loss_rate = 0.01
# 优化器
optimizer = torch.optim.SGD(net.parameters(), loss_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 训练轮次数
epoch = 40

# 添加tensorboard
writer = SummaryWriter(log_dir='logs/lr_001_seed123_ep40')
best_val_accuracy = -1.0
for i in range(epoch):
    print("-----------第{}轮训练开始----------".format(i+1))

    #训练步骤开始
    # 不一定非要将网络设置为train才能开始训练
    # 只有一些特定的网络层必须要进入训练模式才可以训练，比如dropout层
    net.train()
    total_train_loss = 0.0
    total_train_correct = 0

    # 训练循环
    for data in train_loader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)
        total_train_loss += loss.item() * imgs.size(0)
        total_train_correct += (outputs.argmax(dim=1) == targets).sum().item()
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数{}, loss:{} ".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    train_loss_epoch = total_train_loss / train_data_size
    train_accuracy_epoch = total_train_correct / train_data_size

    print(f"训练集平均 Loss：{train_loss_epoch:.4f}")
    print(f"训练集准确率：{train_accuracy_epoch:.2%}")
    writer.add_scalar("train_loss_epoch", train_loss_epoch, i + 1)
    writer.add_scalar("train_accuracy_epoch", train_accuracy_epoch, i + 1)

    # 每轮训练结束后，用验证集检查当前模型
    net.eval()
    total_val_loss = 0.0
    total_val_correct = 0

    # 测试循环
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)

            total_val_loss += loss.item() * imgs.size(0)
            total_val_correct += (
                outputs.argmax(dim=1) == targets
            ).sum().item()

    val_loss = total_val_loss / val_data_size
    val_accuracy = total_val_correct / val_data_size

    print(f"验证集平均 Loss：{val_loss:.4f}")
    print(f"验证集准确率：{val_accuracy:.2%}")
    writer.add_scalar("val_loss", val_loss, i + 1)
    writer.add_scalar("val_accuracy", val_accuracy, i + 1)
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(net.state_dict(), "CIFAR10_lr001_seed123_ep40.pth")
        print(f"第{i + 1}轮刷新最佳验证准确率：{best_val_accuracy:.2%}，已保存参数")

# net.load_state_dict(torch.load(
#     "CIFAR10_lr001_seed123.pth",
#     map_location="cpu",
#     weights_only=True
# ))
# net.eval()
# total_test_loss = 0.0
# total_test_correct = 0

# with torch.no_grad():
#     for imgs, targets in test_loader:
#         imgs, targets = imgs.to(device), targets.to(device)
#         outputs = net(imgs)
#         loss = loss_fn(outputs, targets)
#         total_test_loss += loss.item() * imgs.size(0)
#         total_test_correct += (outputs.argmax(dim=1) == targets).sum().item()

# test_loss = total_test_loss / test_data_size
# test_accuracy = total_test_correct / test_data_size
# print(f"测试集平均 Loss：{test_loss:.4f}")
# print(f"测试集准确率：{test_accuracy:.2%}")

writer.close()






