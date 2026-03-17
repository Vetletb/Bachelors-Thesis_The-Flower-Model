from gabor_like.external._gabor_like_filters import _gabor_like_filters
import torch
from torch.nn.functional import normalize


def _create_filterbank(
    N: int, sigma: float, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    filters = _gabor_like_filters(N, sigma)

    filters = filters.astype("complex64")
    filters = torch.tensor(filters, device=device)

    abs_filters = torch.abs(filters)

    filter_amount = filters.size(dim=0)
    filters = filters.view(filter_amount, -1)
    abs_filters = abs_filters.view(filter_amount, -1)

    filters_list = []

    filters_list.append(filters.real)
    filters_list.append(filters.imag)

    filters = torch.vstack(filters_list)

    filters_normalized = normalize(filters, dim=1)

    return filters_normalized, abs_filters


def _shift_filters(filters: torch.Tensor, steps: int, img_res: int) -> torch.Tensor:
    filter_amount = filters.size(dim=0)
    even_img = img_res % 2 == 0
    filter_radius = (filters.size(dim=1) - 1) / 2
    if even_img:
        box_size = int(steps * 2 + filter_radius * 2)
        offset = 0
    else:
        box_size = int(steps * 2 + 1 + filter_radius * 2)
        offset = 1

    padded_filters = torch.zeros(
        filter_amount, box_size, box_size, device=filters.device, dtype=filters.dtype
    )
    padded_filters[:, : filters.size(dim=1), : filters.size(dim=1)] = filters

    filters_list = [padded_filters]

    for current_h_step in range(1, steps * 2 + offset):
        right_shift = torch.roll(padded_filters, shifts=current_h_step, dims=2)
        filters_list.append(right_shift)

    for current_v_step in range(1, steps * 2 + offset):
        down_shift = torch.roll(padded_filters, shifts=current_v_step, dims=1)
        filters_list.append(down_shift)

        for current_h_step in range(1, steps * 2 + offset):
            down_right_shift = torch.roll(down_shift, shifts=current_h_step, dims=2)
            filters_list.append(down_right_shift)

    return torch.vstack(filters_list)
