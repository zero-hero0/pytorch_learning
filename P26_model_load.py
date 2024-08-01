import torch
import torchvision
from torch import nn
from P26_model_save import Tudui

vgg16_1 = torch.load("vgg16_method1.pth")
# print(vgg16_1)

vgg16_2 = torchvision.models.vgg16(pretrained=False)
vgg16_2.load_state_dict(torch.load("vgg16_method2.pth")) # 存储方式不同加载的方式也不同
# print(vgg16_2)

# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, 3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

model = torch.load("vgg16_method3.pth")
print(model)