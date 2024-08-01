import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from P27_model import *

device = torch.device("cpu")
# torch.device() 是 PyTorch 库中的一个函数，用于创建一个设备对象，用于张量计算。
train_dataset = torchvision.datasets.CIFAR10("cifdataset", train=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10("cifdataset", train=True, transform=torchvision.transforms.ToTensor())

print("the size of train_dataset: {}".format(len(train_dataset)))
print("the size of test_dataset: {}".format(len(test_dataset)))

train_dataloader = DataLoader(train_dataset, 64)
test_dataloader = DataLoader(test_dataset, 64)

tudui = Tudui()
tudui = tudui.to(device) # 有三种东西可以加上to(device)用gpu运行，其中之一是神经网络
# if torch.cuda.is_available():
    # tudui = tudui.cuda()

loss_fn = CrossEntropyLoss()
loss_fn = loss_fn.to(device) # 损失函数可以用gpu
# if torch.cuda.is_available():
#     loss_fn = loss_fn.cuda()

learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr = learning_rate)

total_training_step = 0
total_testing_step = 0

epoch = 10

writer = SummaryWriter("logs")

for i in range(epoch):
    print("the {}th epoch".format(i + 1))

    tudui.train() #  PyTorch 中的一个方法，用于将模型设置为训练模式。在训练模式下，模型可能会使用一些特定的层和激活函数，如 Dropout 和 Batch Normalization。

    for data in train_dataloader:
        img, target = data
        img = img.to(device) # 数据可以用gpu
        target = target.to(device)
        # if torch.cuda.is_available():
        #     img = img.cuda()
        #     target = target.cuda()
        ouput = tudui(img)
        loss = loss_fn(ouput, target)

        optimizer.zero_grad()
        loss.backward() #在深度学习中，通常会在反向传播过程中调用 loss.backward()，以计算损失函数关于模型参数的梯度。这些梯度将用于更新模型参数，以减少损失函数值。loss.backward() 方法不需要任何参数
        optimizer.step()

        total_training_step += 1
        writer.add_scalar("train_error", loss, total_training_step)
        if total_training_step % 100 == 0:
            print("the {}th train, Loss:{}".format(total_training_step, loss.item()))

    total_test_loss = 0
    total_test_accuracy = 0

    tudui.eval() #在评估模式下，模型可能会使用一些特定的层和激活函数，如 Average Pooling 和 Sigmoid。

    with torch.no_grad():
        #在 PyTorch 中，张量（Tensor）在进行某些操作时可能会自动计算梯度。然而，在某些情况下，可能希望临时禁用梯度计算，以提高计算效率。torch.no_grad() 上下文管理器可以在这样的情况下发挥作用。
        for data in test_dataloader:
            imgs, targets = data
            img = img.to(device)
            target = target.to(device)
            # if torch.cuda.is_available():
            #     img = img.cuda()
            #     target = target.cuda()
            output = tudui(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item() #加上item()防止loss有时候不是数值类型而是tensor之类的别的数据类型
            accuracy = (output.argmax(1) == targets).sum() # 用argmax取出最有可能的数据类别，和targets对比
            total_test_accuracy += accuracy

        print("now error in test_data:{}".format(total_test_loss))
        print("now accuracy in test data:{}".format(total_test_accuracy/len(test_dataset)))
        total_testing_step += 1
        writer.add_scalar("test_error", total_test_loss, total_testing_step)
        writer.add_scalar("test_accuracy", total_test_accuracy/len(test_dataset), total_testing_step)

    torch.save(tudui, "tudui_{}.pth".format(i))
    print("the model has been saved")

writer.close()



