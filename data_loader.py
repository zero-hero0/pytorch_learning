import torchvision # torchvision是pytorch库的一个拓展库，主要用于计算机视觉任务，例如图像分类，目标检测，
# 提供了COCO,VOC等数据集加载方式，提供了VGG，ResNet等模型的实现，提供了图像处理和转换的实用功能
from torch.utils.data import DataLoader
# torch.utils 是 PyTorch 库的一部分，它包含了一些工具和实用函数，用于加速模型的训练和优化模型的性能。torch.utils 中有很多有用的功能，例如数据加载、模型权重初始化、模型平均等
from torch.utils.tensorboard import SummaryWriter
#torch.utils.tensorboard 是 PyTorch 库中的一个模块，用于实现 TensorBoard，一个用于可视化训练过程中 tensor 变化的工具
# TensorBoard 可以实时查看训练过程中的数据，如损失函数、指标等，这有助于理解模型训练的过程。
test_data = torchvision.datasets.CIFAR10("./cifdataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
#batch_size=64：指定每个小批量的样本数量为 64。
#shuffle=True：指定是否打乱数据顺序，这里设置为 True，表示打乱数据顺序。
# num_workers=0：指定用于数据加载的线程数量为 0，这里设置为 0 表示使用主线程加载数据。
# drop_last=True：指定是否抛弃最后一个不完整的批量，这里设置为 True，表示抛弃最后一个不完整的批量。

# img, target = test_data[0]
# print(img.shape)
# print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step) # 注意是add_images而不是add_image
#writer.add_images() 函数的参数如下：
# tag：一个字符串，用于为图像指定一个标签。在 TensorBoard 中，可以通过这个标签来区分和筛选图像。
# imgs：一个张量（Tensor），表示要添加的图像。这个张量应该是一个 4 维张量，形状为 (N, C, H, W)，其中 N 是图像数量，C 是图像通道数，H 和 W 是图像的高度和宽度。
# global_step：一个整数，表示当前的训练步数。在 TensorBoard 中，可以通过这个步数来区分和排序图像。

        step += 1

writer.close()