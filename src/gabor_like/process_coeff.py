import os
import torch

from gabor_like.external._gabor_like_filters import _gabor_like_filters
from gabor_like._kmeans import _spherical_kmeans


class CoeffProcessor:
    def __init__(
        self,
        path: str,
        img_res: int,
        sigma: float,
        k: int
    ):
        self.coeff_path = os.path.join(path, "extracted_features")
        self.num_batches = len(os.listdir(self.coeff_path))
        self.img_res = img_res
        self.sigma = sigma
        self.k = k

        # Select device to run tensors on: CPU or CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process(self):
        if self.img_res % 2 == 0:
            N = self.img_res - 1
        else:
            N = self.img_res
        
        filters = _gabor_like_filters(N, self.sigma)
        filters = filters.astype("complex64")
        filters = torch.tensor(filters, device=self.device)
        abs_filters = torch.abs(filters)
        abs_filters = abs_filters.view(abs_filters.shape(0), -1)
        
        cluster_labels = _spherical_kmeans(self.k, abs_filters, 1000)

        for i in range(self.num_batches):
            current_batch = torch.load(os.path.join(self.coeff_path, f"{i}.pt"))
