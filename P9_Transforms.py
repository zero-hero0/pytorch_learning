from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img_path = "data/train/ants_image/24335309_c5ea483bb8.jpg"
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)


writer.add_image("tensor_image", img_tensor)
writer.close()
