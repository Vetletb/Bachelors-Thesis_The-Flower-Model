from .external._gabor_like_filters import _gabor_like_filters
import torch
from torch.nn.functional import normalize


def _create_filterbank(
    N: int, sigma: float, device: str, steps: int, percent: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create normalized filterbank split into real and imaginary components.

    Generates complex Gabor-like, then separates into real/imag parts.
    Real parts interleaved with imaginary parts in output.

    Args:
        N: filter size, must be odd and close to image resolution.
        sigma: Sigma used for generating filters.
        device: Torch device for tensor placement.
        steps: How many filter sizes of same rotation gets generated.
        percent: Determines minimum filter size. Between 0-1

    Returns:
        filters_normalized: Shape (2 * num_filters, N * N).
            Real parts stacked above imaginary parts, normalized to unit L2 norm.
        abs_filters: Shape (num_filters, N * N).
            Magnitude of complex filters before splitting.

    Notes:
        Output interleaving is critical: idx 0..n-1 are real parts, n..2n-1 are
        imaginary parts. This matches cosine_similarity indexing for complex
        dot product reconstruction.
    """
    filters, _ = _gabor_like_filters(N, sigma, steps, percent)

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


def _shift_filters(
    filters: torch.Tensor, steps: int, img_res: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a grid of spatially shifted filter copies with position metadata.

    Translates copies of input filters horizontally, vertically, and diagonally
    to simulate a sampling grid across image space. Each unique translation gets
    a unique position (y, x) offset from image.

    Args:
        filters: Shape (num_filters, N, N).
        steps: Half-width of translation grid. How far to move filter from center of image.
        img_res: Input image resolution for computing reference center and bounds.

    Returns:
        shifted_filters: Shape (num_filters * grid_size^2, box_size, box_size).
            Padded filter copies with zero-padding for out-of-bounds positions.
        shifted_pos: Shape (num_filters * grid_size^2, 2).
            (y_offset, x_offset) for each shifted filter, relative to image.
    """
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
    shifted_pos = torch.zeros(
        filter_amount * dim_steps**2, 2, device=filters.device, dtype=filters.dtype
    )

    pos_offset = img_res // 2 - steps

    shifted_filters[:filter_amount, :, :] = padded_filters
    shifted_pos[:filter_amount, 0] = pos_offset
    shifted_pos[:filter_amount, 1] = pos_offset

    for current_h_step in range(1, steps * 2 + offset):
        right_shift = torch.roll(padded_filters, shifts=current_h_step, dims=2)
        start = current_h_step * filter_amount
        end = start + filter_amount
        shifted_filters[start:end, :, :] = right_shift
        shifted_pos[start:end, 0] = current_h_step + pos_offset
        shifted_pos[start:end, 1] = pos_offset

    for current_v_step in range(1, steps * 2 + offset):
        down_shift = torch.roll(padded_filters, shifts=current_v_step, dims=1)
        v_start = current_v_step * dim_steps * filter_amount
        v_end = v_start + filter_amount
        shifted_filters[v_start:v_end, :, :] = down_shift
        shifted_pos[v_start:v_end, 0] = pos_offset
        shifted_pos[v_start:v_end, 1] = current_v_step + pos_offset

        for current_h_step in range(1, steps * 2 + offset):
            down_right_shift = torch.roll(down_shift, shifts=current_h_step, dims=2)
            h_start = current_h_step * filter_amount + v_start
            h_end = h_start + filter_amount
            shifted_filters[h_start:h_end, :, :] = down_right_shift
            shifted_pos[h_start:h_end, 0] = current_h_step + pos_offset
            shifted_pos[h_start:h_end, 1] = current_v_step + pos_offset

    return shifted_filters, shifted_pos
