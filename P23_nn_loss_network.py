import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("cifdataset", download=True, train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
# nn.CrossEntropyLoss() 是 PyTorch 库中的一个损失函数类，用于计算交叉熵损失（Cross Entropy Loss），通常用于多分类问题。
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
# torch.optim.SGD 是 PyTorch 库中的一个优化器类，用于实现随机梯度下降（Stochastic Gradient Descent，SGD）算法。
# torch.optim.SGD 类的构造函数有两个参数：
#
# params：一个迭代器，用于返回模型中的参数。通常使用 model.parameters() 获取模型参数。
#
# lr：一个浮点数，表示学习率（learning rate）。学习率决定了每次参数更新时步长的规模。较大的学习率可能会导致参数在较短的时间内快速下降，但可能会导致收敛不稳定；较小的学习率可能会导致参数更新缓慢，从而导致收敛速度较慢。
for epoch in range(20):
    running_loss = 0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)

        result_loss = loss(outputs, targets)
        running_loss += result_loss

        optim.zero_grad()
        # 在深度学习中，通常会在进行参数更新之前调用 optim.zero_grad()，以避免梯度累加。梯度累加是指在多次参数更新之间，梯度值会被重复计算并累加，这可能导致参数更新不稳定。
        optim.step()
        result_loss.backward()

    print(running_loss)



