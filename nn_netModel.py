import os

from torch import nn

os.environ["TORCH_HOME"] = r"D:\AI\torch_cache"

import torch
import torchvision

# train_data = torchvision.datasets.ImageNet(r"D:\AI\dataset", split="train", transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

# 想在vgg模型里加一层
# vgg16_true.add_module('add_linear', nn.Linear(in_features=1000, out_features=10))、

# 想在vgg模型的classifier（Sequential）中加个线性层
# vgg16_true.classifier.add_module('add_linear', nn.Linear(in_features=1000, out_features=10))
print(vgg16_true)

# 我不想加一层，我想修改最后一层
vgg16_false.classifier[6] = nn.Linear(4096, 10)
