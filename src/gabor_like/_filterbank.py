from gabor_like.external._gabor_like_filters import _gabor_like_filters
from gabor_like._kmeans import _spherical_kmeans
import torch
from torch.nn.functional import normalize

SPH_KMEANS_ITERS = 1000


def _create_filterbank(
    N: int, sigma: float, k: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _shift_filters(filters: torch.Tensor, steps: int) -> torch.Tensor:
    padding = torch.nn.ZeroPad2d(steps)
    padded_filters = padding(filters)

    filters_list = [padded_filters]

    for current_v_step in range(1, steps + 1):
        up_shift = torch.roll(padded_filters, shifts=current_v_step, dims=2)
        down_shift = torch.roll(padded_filters, shifts=-current_v_step, dims=2)
        filters_list.append(up_shift)
        filters_list.append(down_shift)

    for current_h_step in range(1, steps + 1):
        left_shift = torch.roll(padded_filters, shifts=current_h_step, dims=1)
        right_shift = torch.roll(padded_filters, shifts=-current_h_step, dims=1)
        filters_list.append(left_shift)
        filters_list.append(right_shift)

        for current_v_step in range(1, steps + 1):
            left_up_shift = torch.roll(left_shift, shifts=current_v_step, dims=2)
            right_up_shift = torch.roll(right_shift, shifts=current_v_step, dims=2)
            filters_list.append(left_up_shift)
            filters_list.append(right_up_shift)

            left_down_shift = torch.roll(left_shift, shifts=-current_v_step, dims=2)
            right_down_shift = torch.roll(right_shift, shifts=-current_v_step, dims=2)
            filters_list.append(left_down_shift)
            filters_list.append(right_down_shift)

    return torch.vstack(filters_list)
