import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from CIFAR10_model import Net

test_data = datasets.CIFAR10(
    root=r"D:\AI\dataset",
    train=False,
    download=False,
    transform=transforms.ToTensor()
)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)

checkpoint = r"D:\AI\projects\torchVision\CIFAR10_lr001_seed123_ep40.pth"
net.load_state_dict(torch.load(
    checkpoint, map_location=device, weights_only=True
))
net.eval()

loss_fn = nn.CrossEntropyLoss()
total_loss = 0.0
total_correct = 0

with torch.no_grad():
    for imgs, targets in test_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = net(imgs)

        total_loss += loss_fn(outputs, targets).item() * imgs.size(0)
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()

print("测试模型：", checkpoint)
print("测试样本数：", len(test_data))
print(f"测试集平均 Loss：{total_loss / len(test_data):.4f}")
print(f"测试集准确率：{total_correct / len(test_data):.2%}")