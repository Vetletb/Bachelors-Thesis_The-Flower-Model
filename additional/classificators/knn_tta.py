import torch
from coeff_dataset import FeatureDataset

img_res = 64
sigma = 50.
batch_size = 10
steps = 8
percent = 0.7

k = 1000
k_around_eye = 100
low_freq_upper= 6.75
high_freq_lower = 14.
aug_amount_train = 1000
aug_amount_test = 1000

device = "cuda"

memory_safe = True

USERNAME = ''
DATASET = 'dataset'
COEFF_PATH = f'/cluster/work/{USERNAME}/dataset/coeff_{img_res}_{sigma}_{steps}_{percent}/'
TRAIN_PATH  = f'/cluster/work/{USERNAME}/dataset/coeff_{img_res}_{sigma}_{steps}_{percent}/train/maxpool_{k}_{k_around_eye}_{low_freq_upper}_{high_freq_lower}_{aug_amount_train}'
TEST_PATH = f'/cluster/work/{USERNAME}/dataset/coeff_{img_res}_{sigma}_{steps}_{percent}/test/maxpool_{k}_{k_around_eye}_{low_freq_upper}_{high_freq_lower}_{aug_amount_test}'

# Load train and test datasets
print("dataset")
train_dataset = FeatureDataset(COEFF_PATH, TRAIN_PATH)
test_dataset = FeatureDataset(COEFF_PATH, TEST_PATH)
print("done")
X_train = train_dataset.features
y_train = train_dataset.labels

X_test = test_dataset.features
y_test = test_dataset.labels

# Prepare train data
train_images = X_train.size(dim=0)
train_augments = X_train.size(dim=1)
train_coeffs = X_train.size(dim=2)

X_train = X_train.view(-1, k_around_eye*3)
y_train = y_train.repeat_interleave(train_augments)

X_train = X_train.to(device)
y_train = y_train.to(device)


# Prepare test data
test_images = X_test.size(dim=0)
test_augments = X_test.size(dim=1)
test_coeffs = X_test.size(dim=2)

X_test = X_test.view(-1, k_around_eye*3)

X_test = X_test.to(device)
y_test = y_test.to(device)

if not memory_safe:

    # knn-tta
    distances = torch.cdist(X_test, X_train)
    # print(distances)

    min_indices = torch.argmin(distances.view(test_images, -1), dim=1)
    # print(min_indices)

    train_indices = min_indices % (train_augments * train_images)
    # print(train_indices)

    predicted_labels = y_train[train_indices]
    # print(predicted_labels)

    hit = predicted_labels == y_test
    # print(hit)

    score = hit.float().mean()
    print(score)

else:
    # knn-tta (one test image at a time)
    predicted_labels = torch.empty(test_images, dtype=y_train.dtype, device=device)

    for i in range(test_images):
        print(i)
        start = i * test_augments
        end = (i + 1) * test_augments

        distances_i = torch.cdist(X_test[start:end], X_train)

        min_idx = torch.argmin(distances_i.view(-1))

        train_idx = min_idx % (train_augments * train_images)

        predicted_labels[i] = y_train[train_idx]

        del distances_i, min_idx, train_idx

    hit = predicted_labels == y_test
    score = hit.float().mean()
    print(score)