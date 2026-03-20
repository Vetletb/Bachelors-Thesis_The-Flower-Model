import os
import torch

from .external._gabor_like_filters import _gabor_like_filters
from ._kmeans import _spherical_kmeans
from ._filterbank import _shift_filters
from ._data import _save_pooled, _prepare_folder
from ._data import PREPROCESSEDS_FOLDER

SPH_KMEANS_ITERS = 100


class CoeffProcessor:
    def __init__(
        self, path: str, img_res: int, sigma: float, k: int, f_steps: int, f_percent
    ):
        self.f_steps = f_steps
        self.f_percent = f_percent
        self.path = path
        coeff_path = os.path.join(path, PREPROCESSEDS_FOLDER)
        self.num_batches = len(os.listdir(coeff_path))
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

        filters = _gabor_like_filters(N, self.sigma, self.f_steps, self.f_percent)
        filters = filters.astype("complex64")
        filters = torch.tensor(filters, device=self.device)

        filters = filters.abs()

        filters_amount = filters.size(dim=0)

        filters = _shift_filters(filters, steps, self.img_res)
        filters = filters.view(filters.size(dim=0), -1)

        shift_width = 2 * steps if self.img_res % 2 == 0 else (2 * steps + 1)
        shifted_filters_amount = filters.size(dim=0)

        cluster_labels = _spherical_kmeans(self.k, filters, SPH_KMEANS_ITERS)

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

        pool_path = os.path.join(self.path, f"maxpool_{steps}_{self.k}")
        _prepare_folder(pool_path)

        for i in range(self.num_batches):
            print(i)
            current_batch = torch.load(os.path.join(self.coeff_path, f"{i}.pt"))
            current_batch = current_batch.to(self.device)
            current_batch = current_batch.abs()

            current_batch_size = current_batch.size(dim=0)
            current_batch = current_batch.view(
                current_batch_size, -1
            )  # (images, coeff)

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

            _save_pooled(max_coeff.cpu(), i, pool_path)
            i += 1
