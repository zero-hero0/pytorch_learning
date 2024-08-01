from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img_path = "data/train/ants_image/24335309_c5ea483bb8.jpg"
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
# transforms.ToTensor() 是 PyTorch 中的一个转换（transform）函数，用于将图像（PIL 图像或 OpenCV 图像）转换为张量（Tensor）。
img_tensor = tensor_trans(img)

writer.add_image("tensor_image", img_tensor)
writer.close()
