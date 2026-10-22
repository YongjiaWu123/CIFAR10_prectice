import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
writer = SummaryWriter(log_dir='./logs')

transform = transforms.ToTensor()

train_set = torchvision.datasets.CIFAR10(
    # 要想表达路径，则字符串之前可以加一个r，表达windows路径
    root=r"D:\AI\dataset",
    train=True,
    transform=transform,
    download=True
)
# print(train_set)
#
# print("数据数量：", len(train_set))

# img, target = train_set[0]


# print("图片类型：", type(img))
# print("图片形状：", img.shape)
# print("标签编号：", target)
# print("所有类别：", train_set.classes)
# print("图片类别：", train_set.classes[target])

# Test
# for i in range(10):
#     img, target = train_set[i]
#     writer.add_image("CIFAR10_img", img, i)



train_loader = DataLoader(
    #创建哪个数据集的dataloader
    dataset=train_set,
    #一批有多少个数据
    batch_size=64,
    #是否打乱顺序
    shuffle=True
)

for imgs, targets in train_loader:
    print("一批图片的形状：", imgs.shape)
    print("一批标签的形状：", targets.shape)
    print("这一批的标签：", targets)
    break

