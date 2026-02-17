from gabor_like.external._gabor_like_filters import _gabor_like_filters
from gabor_like._kmeans import _spherical_kmeans
import torch
from torch.nn.functional import normalize

SPH_KMEANS_ITERS = 1000

def _create_filterbank(N: int, sigma: float, k: int, device: str) -> torch.Tensor:
    filters, abs_filters, sum_abs_filters = _gabor_like_filters(N, sigma)

    filters = filters.astype("complex64")
    abs_filters = abs_filters.astype("float32")
    sum_abs_filters = sum_abs_filters.astype("float32")

    sum_abs_filters = torch.tensor(sum_abs_filters, device=device)
    abs_filters = torch.tensor(abs_filters, device=device)
    filters = torch.tensor(filters, device=device)

    filter_amount = filters.size(dim=0)
    filters = filters.view(filter_amount, -1)
    abs_filters = abs_filters.view(filter_amount, -1)

    labels = _spherical_kmeans(k, abs_filters, SPH_KMEANS_ITERS)

    max_filter_list = []
    for i in range(k):
        filters_in_cluster = filters[labels == i]
        max_filter_real = torch.amax(filters_in_cluster.real, dim=0)
        max_filter_imag = torch.amax(filters_in_cluster.imag, dim=0)  
        max_filter_list.append(max_filter_real)
        max_filter_list.append(max_filter_imag)

    max_filters = torch.stack(max_filter_list)

    filters_list = []

    filters_list.append(max_filters)
    filters_list.append(max_filters * -1)

    filters = torch.vstack(filters_list)

    filters_normalized = normalize(filters, dim=1)

    return filters_normalized, sum_abs_filters
