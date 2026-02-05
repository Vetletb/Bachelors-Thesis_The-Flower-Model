from torchvision import transforms 
from torchvision.datasets import ImageFolder
from PIL import Image
from external.filterbank import filterbank
from src.cosine_similarity import cosine_similarity
from src.data import get_dataloader
from src.filterbank import get_filterbank
import torch

PATH = "dataset/train"
IMG_RES = 32
SIGMA = 20.
BATCH_SIZE = 1

loader = get_dataloader(path=PATH, batch_size=BATCH_SIZE, img_res=IMG_RES)
images, labels = next(iter(loader))

def image_to_tensor(file_name: str):
    img = Image.open(file_name).convert('L')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
        ])

    return transform(img)

def unfold_img(img) -> torch.Tensor:
    unfold = torch.nn.Unfold(kernel_size=(31, 31), padding=15)
    return unfold(img)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
    ])
dataset = ImageFolder(root="dataset/train", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
images, labels = next(iter(loader))

unfolded_img = unfold_img(images)

arr1, arr2, frequency_domain_sum = filterbank()

img = image_to_tensor("dataset/train/airplane/0000.jpg")

filter = torch.Tensor(arr2[0].real)
filter_norm = torch.linalg.vector_norm(filter)
filter_normalized = filter / filter_norm

features = cosine_similarity(filter_normalized, unfolded_img, torch.Tensor(frequency_domain_sum))