from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "data/train/bees_image/16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)
# img_PIL.show()
img_array = np.array(img_PIL) # 图片的格式要为numpy数组或者tensor, 所以要改变格式
print(type(img_array))
print(img_array.shape)

# writer.add_image("train", img_array, 2, dataformats='HWC')
writer.add_image("test", img_array, 1, dataformats='HWC')
#dataformats：一个字符串，用于指定图像数据的格式。在这里，HWC 表示图像数据的格式为高度（H）×宽度（W）×通道（C）。

# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)

writer.close()
