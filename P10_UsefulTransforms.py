from PIL import  Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("image/R-C.png")

#ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

#Normalize
print(img_tensor[0][0][0])
trans_tonorm = transforms.Normalize([3, 1, 6], [3, 4, 1])
img_norm = trans_tonorm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)

#Resize
print(img.size)
trans_toresize = transforms.Resize((512, 512))
img_resize = trans_toresize(img)
print(img_resize)
img_resize_tensor = trans_totensor(img_resize)
writer.add_image("Resize", img_resize_tensor, 0)

# Compose - resize -2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop((100, 100))
trans_compos_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    trans_crop = trans_compos_2(img)
    writer.add_image("RandomCrop", trans_crop, i)

writer.close()
