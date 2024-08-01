import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

dataset = torchvision.datasets.CIFAR10("cifdataset", train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        x = self.maxpool1(x)
        return x

input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)

tudui = Tudui()
output = tudui(input)

# print(output)
# ceil_mode = True
# tensor([[[[2., 3.],
#           [5., 1.]]]])

# ceil_mode = False
# tensor([[[[2.]]]])
# 最大池化不改变通道数

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, traget = data
    output = tudui(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", imgs, step)
    step += 1

writer.close()