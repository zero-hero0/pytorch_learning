import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./cifdataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./cifdataset", train=False, transform=dataset_transform, download=True)

writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i] #之所以可以直接对类用索引是因为Dadaset的__getitem__方法
    writer.add_image("test_set", img, i)