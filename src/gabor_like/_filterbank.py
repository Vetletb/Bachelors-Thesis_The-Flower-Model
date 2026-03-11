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


def _shift_filters(N: int, filters: torch.Tensor, device: str):
    steps = 3
    step_size = int((N - 1) / (2 * steps) * 0.2)

    filters_amount = filters.size(dim=0)

    filters_list = [filters]

    current_v_step = 0
    for _ in range(1, steps+1):
        current_v_step += step_size

        up_shift = torch.cat(
            [
                filters[:, current_v_step:],
                torch.zeros(
                    (filters_amount, current_v_step, N),
                    dtype=filters.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        filters_list.append(up_shift)

        down_shift = torch.cat(
            [
                torch.zeros(
                    (filters_amount, current_v_step, N),
                    dtype=filters.dtype,
                    device=device,
                ),
                filters[:, :-current_v_step],
            ],
            dim=1,
        )
        filters_list.append(down_shift)

    current_h_step = 0
    for _ in range(1, steps+1):
        current_h_step += step_size

        left_shift = torch.cat(
            [
                filters[:, :, current_h_step:],
                torch.zeros(
                    (filters_amount, N, current_h_step),
                    dtype=filters.dtype,
                    device=device,
                ),
            ],
            dim=2,
        )
        filters_list.append(left_shift)

        right_shift = torch.cat(
            [
                torch.zeros(
                    (filters_amount, N, current_h_step),
                    dtype=filters.dtype,
                    device=device,
                ),
                filters[:, :, :-current_h_step],
            ],
            dim=2,
        )
        filters_list.append(right_shift)

        current_v_step = 0
        for _ in range(1, steps+2):
            current_v_step += step_size

            left_up_shift = torch.cat(
                [
                    left_shift[:, current_v_step:],
                    torch.zeros(
                        (filters_amount, current_v_step, N),
                        dtype=left_shift.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            right_up_shift = torch.cat(
                [
                    right_shift[:, current_v_step:],
                    torch.zeros(
                        (filters_amount, current_v_step, N),
                        dtype=right_shift.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            filters_list.append(left_up_shift)
            filters_list.append(right_up_shift)

            left_down_shift = torch.cat(
                [
                    torch.zeros(
                        (filters_amount, current_v_step, N),
                        dtype=left_shift.dtype,
                        device=device,
                    ),
                    left_shift[:, :-current_v_step],
                ],
                dim=1,
            )
            right_down_shift = torch.cat(
                [
                    torch.zeros(
                        (filters_amount, current_v_step, N),
                        dtype=right_shift.dtype,
                        device=device,
                    ),
                    right_shift[:, :-current_v_step],
                ],
                dim=1,
            )
            filters_list.append(left_down_shift)
            filters_list.append(right_down_shift)

    return torch.vstack(filters_list)
