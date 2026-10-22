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

train_loader = DataLoader(
    dataset=train_set,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    drop_last=False
)

print("总样本数：", len(train_set))
print("总批次数：", len(train_loader))

step = 0
writer = SummaryWriter(log_dir="./logs")
for images, labels in train_loader:
    for img in images:
        writer.add_image(
            "img_tensor_oneBy",
            img,
            global_step=step
        )
        print("正在记录 Step:", step)
        step += 1
        if step >= 20:
            break
    if step >= 20:
        break
    # writer.add_images("images", images, step)
writer.close()
