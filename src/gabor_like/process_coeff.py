import os
import torch

from .external._gabor_like_filters import _gabor_like_filters
from ._clustering import _spherical_kmeans
from ._filterbank import _shift_filters
from ._data import _save_pooled, _prepare_folder
from ._data import PREPROCESSED_FOLDER

SPH_KMEANS_ITERS = 25


class CoeffProcessor:
    """
    Process saved coefficients using filter-cluster pooling.

    Args:
        path: Root directory containing preprocessed coefficient folder.
        img_res: Square image resolution used in preprocessing.
        sigma: Sigma used for generating filters.
        f_steps: How many filter sizes of same rotation gets generated.
        f_percent: Determines minimum filter size. Between 0-1.

    Notes:
        sigma, f_steps and f_percent needs to be the same as used for preprocessor.
    """
    def __init__(
        self,
        path: str,
        img_res: int,
        sigma: float,
        f_steps: int,
        f_percent: float,
    ):
        self.f_steps = f_steps
        self.f_percent = f_percent
        self.path = path
        self.coeff_path = os.path.join(path, PREPROCESSED_FOLDER)
        self.num_batches = len(os.listdir(self.coeff_path))
        self.img_res = img_res
        self.sigma = sigma

        # Select device to run tensors on: CPU or CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process(self, k_around_eye: int, low_freq_upper: float, high_freq_lower: float):
        """
        Max-pool coefficients from center-near clusters across frequency bands. Saves result in `path`

        Chooses `k_around_eye` clusters nearest image center for each band:
        low frequency, mid frequency, high frequency.

        Args:
            k_around_eye: Number of nearest clusters to keep per frequency band.
            low_freq_upper: Upper threshold for low-frequency centroid radius.
            high_freq_lower: Lower threshold for high-frequency centroid radius.
        """
        self.k_around_eye = k_around_eye
        self.low_freq_upper = low_freq_upper
        self.high_freq_lower = high_freq_lower

        eye = self._eye_labels()

        # repeat cluster labels to match coefficients, one complex filter gives two filters (real/imag)
        cluster_labels_exp = self.cluster_labels.view(
            -1, self.img_res * self.img_res
        ).repeat(1, 2).view(-1)

        pool_path = os.path.join(
            self.path,
            f"maxpool_{self.k}_{k_around_eye}_{low_freq_upper}_{high_freq_lower}",
        )
        _prepare_folder(pool_path)

        for i in range(self.num_batches):
            print(i)
            # Load the preprocessed coeffs
            current_batch = torch.load(os.path.join(self.coeff_path, f"{i}.pt"))
            current_batch = current_batch.to(self.device)
            # Take absolute values to account for inverse filters
            current_batch = current_batch.abs()

            current_batch_size = current_batch.size(dim=0)
            current_batch = current_batch.view(
                current_batch_size, -1
            )

            max_coeff = torch.empty(
                (current_batch_size, k_around_eye * 3),
                device=self.device,
            )

            # filter out coeffs in selected cluster and max-pool
            write_idx = 0
            for j in eye:
                mask = cluster_labels_exp == j
                coeff_in_cluster = current_batch[:, mask]
                max_coeff[:, write_idx] = coeff_in_cluster.amax(dim=1)
                write_idx += 1
                del coeff_in_cluster

            _save_pooled(max_coeff.cpu(), i, pool_path)

    def cluster(self, k):
        """
        Cluster filters with spherical k-means to prepare for coefficient pooling.

        Args:
            k: Number of spherical k-means clusters.
        """
        self.k = k

        # Calculate filter size N
        if self.img_res % 2 == 0:
            self.N = self.img_res - 1
        else:
            self.N = self.img_res

        # steps for shifting (shifts across whole image)
        steps = self.img_res // 2

        # Get filters and filter radiuses
        filters, filter_r = _gabor_like_filters(
            self.N, self.sigma, self.f_steps, self.f_percent
        )
        filters = filters.astype("complex64")
        filters = torch.tensor(filters, device=self.device)
        filter_r = torch.tensor(filter_r, device=self.device)

        # Absolute of complex filters for clustering
        filters = filters.abs()

        self.filters_amount = filters.size(dim=0)

        filters, filter_pos = _shift_filters(filters, steps, self.img_res)
        filters = filters.view(filters.size(dim=0), -1)

        # Match filter radiuses with shifted filters
        shift_count = filters.size(dim=0) // self.filters_amount
        filter_r = filter_r.repeat(shift_count)

        self.shifted_filters_amount = filters.size(dim=0)

        cluster_labels = _spherical_kmeans(self.k, filters, SPH_KMEANS_ITERS)

        del filters

        # Reorder to match saved coeffs order
        cluster_labels_list = []
        filter_r_list = []
        filter_pos_list = []
        for i in range(self.filters_amount):
            for j in range(i, self.shifted_filters_amount, self.filters_amount):
                cluster_labels_list.append(cluster_labels[j])
                filter_r_list.append(filter_r[j])
                filter_pos_list.append(filter_pos[j])

        self.cluster_labels = torch.hstack(cluster_labels_list)
        self.filter_r = torch.hstack(filter_r_list)
        self.filter_pos = torch.vstack(filter_pos_list)

    def _eye_labels(self) -> torch.Tensor:
        """
        Select cluster labels nearest image center in each frequency band.

        Returns:
            Tensor of shape (3 * k_around_eye,) containing selected cluster ids
            ordered as low-band labels, then mid-band labels, then high-band labels.
        """

        # Calculate radius and positions of all cluster centroids
        centroid_r_list = []
        centroid_pos_list = []
        for cluster in range(self.k):
            cluster_mask = self.cluster_labels == cluster
            r_in_cluster = self.filter_r[cluster_mask]
            pos_in_cluster = self.filter_pos[cluster_mask]

            if r_in_cluster.numel() == 0:
                centroid_r_list.append(torch.tensor([-1], device=self.device))
                centroid_pos_list.append(torch.tensor([self.img_res*2, self.img_res*2], device = self.device))
            else:
                centroid_r_list.append(torch.mean(r_in_cluster))
                centroid_pos_list.append(torch.mean(pos_in_cluster, dim=0))

        centroids_r = torch.hstack(centroid_r_list)
        centroids_pos = torch.stack(centroid_pos_list)

        # Masks for low, mid and high frequency centroids
        low_freq_mask = (centroids_r >= 0) & (centroids_r < self.low_freq_upper)
        mid_freq_mask = (centroids_r >= self.low_freq_upper) & (
            centroids_r <= self.high_freq_lower
        )
        high_freq_mask = centroids_r > self.high_freq_lower

        # Place eye in center of image and calculate centroid distances
        eye_pos = torch.tensor([self.N / 2, self.N / 2], device=self.device)
        pos_diff = centroids_pos - eye_pos
        centroid_distance = torch.linalg.vector_norm(pos_diff, dim=1)

        # Split the centroid distances for low, mid and high frequencies
        low_freq_distance = centroid_distance[low_freq_mask]
        mid_freq_distance = centroid_distance[mid_freq_mask]
        high_freq_distance = centroid_distance[high_freq_mask]

        # Get indices of centroid distances closest to eye
        low_eye_idx = low_freq_distance.topk(self.k_around_eye, largest=False).indices
        mid_eye_idx = mid_freq_distance.topk(self.k_around_eye, largest=False).indices
        high_eye_idx = high_freq_distance.topk(self.k_around_eye, largest=False).indices

        # Create tensor with cluster ids corresponding to centroid distances
        label_array = torch.arange(0, self.k, device=self.device)

        low_freq_label = label_array[low_freq_mask]
        high_freq_label = label_array[high_freq_mask]
        mid_freq_label = label_array[mid_freq_mask]

        # Select cluster id from eye indexes
        low_eye = low_freq_label[low_eye_idx]
        mid_eye = mid_freq_label[mid_eye_idx]
        high_eye = high_freq_label[high_eye_idx]

        eye_list = [low_eye, mid_eye, high_eye]
        eye = torch.hstack(eye_list)

        return eye
