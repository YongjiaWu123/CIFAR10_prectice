import torch
from torch import nn

input = torch.randn(1, 1, 1, 3)
target = torch.randn(1, 1, 1, 3)

print(input)
print(target)

# reduction = 'mean'为平均数 // 'sum'为和
l1loss = nn.L1Loss(reduction='mean')
l2loss = nn.L1Loss(reduction='sum')
print(l1loss(input, target))
print(l2loss(input, target))

mseloss = nn.MSELoss(reduction='mean')
print(mseloss(input, target))


cross_entropy = nn.CrossEntropyLoss(reduction='mean')

print("交叉熵损失--------------")
cel_output = torch.tensor([0.5, 0.4, 0.8, 0.2])
cel_target = torch.tensor([0])
loss_cel = cross_entropy(cel_output, cel_target)
print(loss_cel)


print("TEST-----")
output_test = torch.tensor([0.1, 0.2, 0.3])
torch.reshape(output_test, (1, 3))
target_test = torch.tensor([1])
loss_cel_test = cross_entropy(output_test, target_test)
print(loss_cel_test)
