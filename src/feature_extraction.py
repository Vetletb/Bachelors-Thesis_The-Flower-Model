from src.cosine_similarity import cosine_similarity
from src.data import get_dataloader
from src.filterbank import get_filterbank
import torch

# Constants
PATH = "dataset/train"
IMG_RES = 32
SIGMA = 20.
BATCH_SIZE = 1
MAX_POOL_SIZE = 2

# Select device to run tensors in: CPU or CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate kernel/filter size, needs to be an odd number close to image size
# Calculate padding necessary for unfolding
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
    images = images.to(device)

    # Extract sliding blocks from batched images
    unfold = torch.nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=padding)
    unfolded_img = unfold(images)

    # Calculate cosine_similarity between unfolded image and filters, result is extracted features
    features = cosine_similarity(filters_normalized, unfolded_img, frequency_domain_sum)
    filter_amount = filters_normalized.size(dim=0)
    current_batch_size = features.size(dim=0)
    features = features.view(current_batch_size, filter_amount, IMG_RES, IMG_RES)

    # Max pool extracted features before saving result
    max_pool = torch.nn.MaxPool2d(kernel_size=MAX_POOL_SIZE)
    max_pooled_img = max_pool(features)
