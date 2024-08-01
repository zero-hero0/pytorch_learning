import torch
from torch import nn
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
#在 PyTorch 中，forward 方法是一个模型的前向传播函数，用于计算输入数据经过模型后的输出。
# 在定义一个神经网络模型时，需要继承 torch.nn.Module 类并实现 forward 方法。forward 方法的输入参数通常是张量（Tensor），
# 表示模型的输入数据。forward 方法应该返回一个张量，表示模型的输出数据
        output = input + 1
        return output

tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)