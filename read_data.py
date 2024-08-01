from torch.utils.data import Dataset
# torch.utils.data.Dataset 是 PyTorch 库中的一个抽象基类，用于表示一个数据集。要使用 Dataset 类，需要继承它并实现 __getitem__ 和 __len__ 方法。
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path) #os.listdir() 是 Python 的 os 模块中的一个函数，用于列出指定目录下的所有文件和子目录的名称。

    def __getitem__(self, idx): #__getitem__ 是 Python 中的一种特殊方法，称为魔术方法（magic method）。当使用 [] 操作符访问类的某个属性或方法时，就会自动调用 __getitem__ 方法。
        #在 PyTorch 的 Dataset 类中，__getitem__ 方法用于从数据集中获取指定索引的样本。在自定义数据集类时，需要重写 __getitem__ 方法以返回正确的样本。
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name) #图片所在位置
        img = Image.open(img_item_path) # 打开图像
        label = self.label_dir #图片所在文件夹的名称就是这张图片的分类
        return img, label

    def __len__(self):
        # 在 PyTorch 的 Dataset 类中，__len__ 方法用于返回数据集中的样本数量。在自定义数据集类时，需要重写 __len__ 方法以返回正确的样本数量。
        return len(self.img_path)

root_dir = "dataset/train" # 数据集所在位置
ants_label_dir = "ants" # 这个数据集小的文件夹名称就是这个文件夹下图片的类型
ants_dataset = MyData(root_dir, ants_label_dir)

bees_label_dir = "bees"
bees_dataset = MyData(root_dir, bees_label_dir)
# img, label = bees_dataset[0]
# img.show()
train_dataset = ants_dataset + bees_dataset
