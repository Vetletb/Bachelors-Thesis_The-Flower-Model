from torchvision import transforms 
from PIL import Image
from src.cosine_similarity import cosine_similarity
from src.data import get_dataloader
from src.filterbank import get_filterbank
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = "dataset/train"
IMG_RES = 32
SIGMA = 20.
BATCH_SIZE = 1

loader = get_dataloader(path=PATH, batch_size=BATCH_SIZE, img_res=IMG_RES)
images, labels = next(iter(loader))

if(IMG_RES % 2 == 0):
    kernel_size = IMG_RES - 1
else:
    kernel_size = IMG_RES
padding = (int) ((kernel_size - 1) / 2)

unfold = torch.nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=padding)
unfolded_img = unfold(images)

filter_normalized, frequency_domain_sum = get_filterbank(N=kernel_size, sigma=SIGMA, device=device)
features = cosine_similarity(filter_normalized, unfolded_img, frequency_domain_sum)
extracted_img = features.reshape(BATCH_SIZE, 1, IMG_RES, IMG_RES)[0][0].cpu()