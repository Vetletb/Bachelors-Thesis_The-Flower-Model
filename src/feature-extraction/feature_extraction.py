from torchvision import transforms 
from PIL import Image

def image_to_tensor(file_name: str):
    img = Image.open(file_name).convert('L')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
        ])

    return transform(img)