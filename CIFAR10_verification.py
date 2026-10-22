import torch
from PIL import Image
import torchvision
from torch import nn

img_path = 'img/ScreenShot_2026-10-21_150615_600.png'
img = Image.open(img_path)
# print(img)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

img = transform(img)
img = torch.reshape(img, (1,3,32,32))
print(img.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
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

model = torch.load(
    "CIFAR10_49_net.pth",
    map_location="cpu",
    weights_only=False
)
print(model)
model.eval()
with torch.no_grad():
    output = model(img)

print(output)
print(output.argmax(dim=1))