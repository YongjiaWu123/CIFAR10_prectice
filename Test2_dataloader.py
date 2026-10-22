import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader

train_set = torchvision.datasets.CIFAR10(
    root=r"D:\AI\dataset",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
# 如果前面有下载过，可以直接用以下语句引入数据集
# test_set = torchvision.datasets.CIFAR10("D:\AI\dataset", train=False, transform=transforms.ToTensor())

train_loader = DataLoader(
    dataset=train_set,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    drop_last=False
)

print("总样本数：", len(train_set))
print("总批次数：", len(train_loader))

# for batch_index, (imgs, targets) in enumerate(train_loader):
#     if batch_index == 0 or batch_index == len(train_loader) - 1:
#         print(
#             f"第 {batch_index + 1} 批：",
#             imgs.shape,
#             targets.shape
#         )
step = 0
writer = SummaryWriter(log_dir="./logs")
for data in train_loader:
    images, labels = data
    for image in images:
        writer.add_image("{}".format(step), image, step + 1)
        step += 1
writer.close()