import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
torch.save(vgg16, "vgg16_method1.pth")

torch.save(vgg16.state_dict(), "vgg16_method2.pth") #只保存参数

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui, "vgg16_method3.pth") #自己创建的神经网络需要类存在才能保存