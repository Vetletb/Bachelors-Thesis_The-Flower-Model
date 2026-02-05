from torchvision import transforms 
from PIL import Image
from external.filterbank import filterbank
from src.cosine_similarity import cosine_similarity
import torch

def image_to_tensor(file_name: str):
    img = Image.open(file_name).convert('L')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
        ])

    return transform(img)

arr1, arr2, frequency_domain_sum = filterbank()

img = image_to_tensor("dataset/train/airplane/0000.jpg")

filter = torch.Tensor(arr2[0].real)
filter_norm = torch.linalg.vector_norm(filter)
filter_normalized = filter / filter_norm

features = cosine_similarity(filter_normalized, img, torch.Tensor(frequency_domain_sum))