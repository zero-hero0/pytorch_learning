import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from P27_model import *

device = torch.device("cpu")

train_dataset = torchvision.datasets.CIFAR10("cifdataset", train=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10("cifdataset", train=True, transform=torchvision.transforms.ToTensor())

print("the size of train_dataset: {}".format(len(train_dataset)))
print("the size of test_dataset: {}".format(len(test_dataset)))

train_dataloader = DataLoader(train_dataset, 64)
test_dataloader = DataLoader(test_dataset, 64)

tudui = Tudui()
tudui = tudui.to(device)
# if torch.cuda.is_available():
    # tudui = tudui.cuda()

loss_fn = CrossEntropyLoss()
loss_fn = loss_fn.to(device)
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

    tudui.train()

    for data in train_dataloader:
        img, target = data
        img = img.to(device)
        target = target.to(device)
        # if torch.cuda.is_available():
        #     img = img.cuda()
        #     target = target.cuda()
        ouput = tudui(img)
        loss = loss_fn(ouput, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_training_step += 1
        writer.add_scalar("train_error", loss, total_training_step)
        if total_training_step % 100 == 0:
            print("the {}th train, Loss:{}".format(total_training_step, loss.item()))

    total_test_loss = 0
    total_test_accuracy = 0

    tudui.eval()

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            img = img.to(device)
            target = target.to(device)
            # if torch.cuda.is_available():
            #     img = img.cuda()
            #     target = target.cuda()
            output = tudui(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_test_accuracy += accuracy

        print("now error in test_data:{}".format(total_test_loss))
        print("now accuracy in test data:{}".format(total_test_accuracy/len(test_dataset)))
        total_testing_step += 1
        writer.add_scalar("test_error", total_test_loss, total_testing_step)
        writer.add_scalar("test_accuracy", total_test_accuracy/len(test_dataset), total_testing_step)

    torch.save(tudui, "tudui_{}.pth".format(i))
    print("the model has been saved")

writer.close()



