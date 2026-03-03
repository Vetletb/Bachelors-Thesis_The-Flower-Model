from gabor_like.external._gabor_like_filters import _gabor_like_filters
from gabor_like._kmeans import _spherical_kmeans
import torch
from torch.nn.functional import normalize

SPH_KMEANS_ITERS = 1000


def _create_filterbank(N: int, sigma: float, k: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    filters = _gabor_like_filters(N, sigma)

    filters = filters.astype("complex64")
    filters = torch.tensor(filters, device=device)

    abs_filters = torch.abs(filters)

    filter_amount = filters.size(dim=0)
    filters = filters.view(filter_amount, -1)
    abs_filters = abs_filters.view(filter_amount, -1)

    labels = _spherical_kmeans(k, abs_filters, SPH_KMEANS_ITERS)

    filters_list = []

    filters_list.append(filters.real)
    filters_list.append(filters.imag)

    filters = torch.vstack(filters_list)

    filters_normalized = normalize(filters, dim=1)

    return filters_normalized, abs_filters, labels
