
import torch
import torch.nn.functional as F
input = torch.tensor([[1,2,0,3,1],
                      [2,3,4,5,6],
                      [3,4,5,6,7],
                      [4,5,6,7,8],
                      [5,6,7,8,9]])
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

# 但是卷积操作对输入的图像有要求，必须除了长宽还要有其他参数‘

print(input.shape)
print(kernel.shape)
input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
print("修改后--------------- ")
print(input.shape)
print(kernel.shape)

output = F.conv2d(input,kernel,stride=1)
print(output)