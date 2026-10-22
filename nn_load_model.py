import torch
import torch.nn
import torchvision

# 方式1->保存方式1，加载模型
# model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
model = torch.load("vgg16_method2.pth")
vgg16.load_state_dict(model)
print(vgg16)