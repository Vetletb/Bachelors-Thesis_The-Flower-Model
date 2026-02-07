from src.cosine_similarity import cosine_similarity
from src.data import create_dataloader, save_result, save_result_labels
from src.filterbank import create_filterbank
import torch
import threading


# Record GPU memory for performance testing
torch.cuda.memory._record_memory_history()

# Constants
PATH = "dataset/train"
IMG_RES = 6
SIGMA = 20.0
BATCH_SIZE = 1000

# Producer consumer variables
cv = threading.Condition()
work_available = False
done = False
results = torch.Tensor()
result_labels = torch.Tensor()


def writer():
    file_index = 0

    while True:
        global work_available, done, results, result_labels
        with cv:
            print()
            while not work_available and not done:
                cv.wait()

            if not work_available and done:
                break

            save_result(results, file_index)
            save_result_labels(result_labels, file_index)

            work_available = False
            print(file_index)
            file_index += 1

            cv.notify_all()


def feature_extractor():
    global work_available, done, results, result_labels

    for images, labels in loader:
        images = images.to(device)

        # Extract sliding blocks from batched images
        unfolded_img = unfold(images)

        # Calculate cosine_similarity between unfolded image and filters, result is extracted features
        features = cosine_similarity(
            filters_normalized, unfolded_img, frequency_domain_sum
        )

        filter_amount = filters_normalized.size(dim=0)
        current_batch_size = features.size(dim=0)
        features = features.view(current_batch_size, filter_amount, IMG_RES, IMG_RES)
        with cv:
            while work_available:
                cv.wait()
            results = features.to("cpu", non_blocking=True).clone()
            result_labels = labels.clone()
            work_available = True
            cv.notify_all()
        del unfolded_img, features
    done = True


# Select device to run tensors in: CPU or CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate kernel/filter size, needs to be an odd number close to image size
# Calculate padding necessary for unfolding
if IMG_RES % 2 == 0:
    kernel_size = IMG_RES - 1
else:
    kernel_size = IMG_RES
padding = (int)((kernel_size - 1) / 2)

# Get the filters as tensors
filters_normalized, frequency_domain_sum = create_filterbank(
    N=kernel_size, sigma=SIGMA, device=device
)

# Create image unfolder
unfold = torch.nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=padding)

# Get a DataLoader object and iterate through the batches
loader = create_dataloader(path=PATH, batch_size=BATCH_SIZE, img_res=IMG_RES)

# Create consumer for writing results
consumer = threading.Thread(target=writer, daemon=True)
consumer.start()

# Start producing extracted features
feature_extractor()

# Wait for consumer to finish writing
consumer.join()

torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
