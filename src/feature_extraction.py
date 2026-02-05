from src.cosine_similarity import cosine_similarity
from src.data import get_dataloader
from src.filterbank import get_filterbank
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = "dataset/train"
IMG_RES = 32
SIGMA = 20.
BATCH_SIZE = 1
MAX_POOL_SIZE = 2

if(IMG_RES % 2 == 0):
    kernel_size = IMG_RES - 1
else:
    kernel_size = IMG_RES
padding = (int) ((kernel_size - 1) / 2)

# Get the filters as tensors
filters_normalized, frequency_domain_sum = get_filterbank(N=kernel_size, sigma=SIGMA, device=device)


# Get a DataLoader object and iterate through the batches 
loader = get_dataloader(path=PATH, batch_size=BATCH_SIZE, img_res=IMG_RES)
for images, labels in loader:

    unfold = torch.nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=padding)
    unfolded_img = unfold(images)

    features = cosine_similarity(filters_normalized, unfolded_img, frequency_domain_sum)
    extracted_img = features.reshape(BATCH_SIZE, 1, IMG_RES, IMG_RES)[0][0].cpu()

    max_pool = torch.nn.MaxPool2d(kernel_size=MAX_POOL_SIZE)
    max_pooled_img = max_pool(features)

    import matplotlib.pyplot as plt
    plt.imshow(extracted_img, cmap="gray")
    plt.axis("off")
    plt.show()