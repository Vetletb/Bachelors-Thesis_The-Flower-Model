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
    N = filters.size(dim=1)

    filter_radius = (N - 1) / 2
    even_img = img_res % 2 == 0
    offset = 0 if even_img else 1
    dim_steps = steps * 2 + offset
    box_size = int(dim_steps + filter_radius * 2)

    padded_filters = torch.zeros(
        filter_amount, box_size, box_size, device=filters.device, dtype=filters.dtype
    )
    padded_filters[:, :N, :N] = filters

    shifted_filters = torch.zeros(
        filter_amount * dim_steps**2,
        box_size,
        box_size,
        device=filters.device,
        dtype=filters.dtype,
    )
    shifted_filters[:filter_amount, :, :] = padded_filters

    for current_h_step in range(1, steps * 2 + offset):
        right_shift = torch.roll(padded_filters, shifts=current_h_step, dims=2)
        start = current_h_step * filter_amount
        end = start + filter_amount
        shifted_filters[start:end, :, :] = right_shift

    current_h_step = 0
    for current_v_step in range(1, steps * 2 + offset):
        down_shift = torch.roll(padded_filters, shifts=current_v_step, dims=1)
        v_start = current_v_step * dim_steps * filter_amount
        v_end = v_start + filter_amount
        shifted_filters[v_start:v_end, :, :] = down_shift

        for current_h_step in range(1, steps * 2 + offset):
            down_right_shift = torch.roll(down_shift, shifts=current_h_step, dims=2)
            h_start = current_h_step * filter_amount + v_start
            h_end = h_start + filter_amount
            shifted_filters[h_start:h_end, :, :] = down_right_shift

    return shifted_filters
