import torchvision
from torch import nn

# dataset = torchvision.datasets.ImageNet("img_net", split="train", download=True, transform=torchvision.transforms.ToTensor())
# 数据集过大，无法下载

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True) #参数训练好的vgg16模型

print(vgg16_true)

# 将全连接层添加到 VGG16 模型的 classifier 部分
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
# 改变classifier部分的第6层
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)