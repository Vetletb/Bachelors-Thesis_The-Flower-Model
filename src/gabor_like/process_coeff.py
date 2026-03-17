import os
import torch

from gabor_like.external._gabor_like_filters import _gabor_like_filters
from gabor_like._kmeans import _spherical_kmeans
from gabor_like._filterbank import _shift_filters


class CoeffProcessor:
    def __init__(self, path: str, img_res: int, sigma: float, k: int):
        self.coeff_path = os.path.join(path, "extracted_features")
        self.num_batches = len(os.listdir(self.coeff_path))
        self.img_res = img_res
        self.sigma = sigma
        self.k = k

        # Select device to run tensors on: CPU or CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process(self, steps):
        if self.img_res % 2 == 0:
            N = self.img_res - 1
        else:
            N = self.img_res

        filters = _gabor_like_filters(N, self.sigma)
        filters = filters.astype("complex64")
        filters = torch.tensor(filters, device=self.device)

        filters_amount = filters.size(dim=0)

        filters = _shift_filters(filters, steps, self.img_res)
        abs_filters = torch.abs(filters)
        abs_filters = abs_filters.view(abs_filters.size(dim=0), -1)

        shift_width = 2 * steps if self.img_res % 2 == 0 else (2 * steps + 1)
        shifted_filters_amount = filters.size(dim=0)

        cluster_labels = _spherical_kmeans(self.k, abs_filters, 100)

        cluster_labels_list = []
        for i in range(filters_amount):
            for j in range(i, shifted_filters_amount, filters_amount):
                cluster_labels_list.append(cluster_labels[j])

        cluster_labels = torch.hstack(cluster_labels_list)

        actual_k = torch.unique(cluster_labels).numel()

        cluster_labels = cluster_labels.view(-1, shift_width, shift_width)
        pad_length = (self.img_res - shift_width) // 2
        padder = torch.nn.ConstantPad2d(pad_length, self.k)
        cluster_labels = padder(cluster_labels)
        cluster_labels = cluster_labels.view(-1, self.img_res * self.img_res)

        cluster_labels_exp = cluster_labels.repeat_interleave(2)  # (labels)

        for i in range(self.num_batches):
            print(i)
            current_batch = torch.load(os.path.join(self.coeff_path, f"{i}.pt"))
            current_batch = current_batch.to(self.device)
            current_batch = current_batch.abs()

            current_batch_size = current_batch.size(dim=0)
            current_batch = current_batch.view(current_batch_size, -1)  # (images, coeff)

            max_coeff = torch.empty(
                (current_batch_size, actual_k),
                device=self.device,
            )

            write_idx = 0
            for j in range(self.k):
                mask = cluster_labels_exp == j
                if mask.sum() == 0:
                    continue
                coeff_in_cluster = current_batch[:, mask]
                max_coeff[:, write_idx] = coeff_in_cluster.amax(dim=1)
                write_idx += 1
                del coeff_in_cluster