import torch


def _cosine_similarity(
    a_normalized: torch.Tensor, b: torch.Tensor, abs: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity between real/imag filters and image patches.

    Performs per-filter dot product via normalized image patches onto filters,
    scaled by filter magnitude for weighted similarity.

    Args:
        a_normalized: Shape (2 * num_filters, spatial_dim).
            Normalized filter components: first num_filters rows are real parts,
            remaining rows are imaginary parts. Must have unit L2 norm along dim=1.
        b: Shape (batch_size, filter_size, num_patches).
            Unfolded image patches. spatial_dim should match a_normalized.
        abs: Shape (num_filters, spatial_dim).
            Magnitude of complex filters before real/imag split. Used to weight
            patch normalization.

    Returns:
        cos_sim: Shape (batch_size, 2 * num_filters, num_patches).
            Cosine similarity scores. Structure
            real_similarity and imag_similarity for filter i for every filter.
    """
    filter_amount = abs.size(dim=0)
    patch_size = b.size(dim=-1)
    batch_size = b.size(dim=0)

    dtype = b.dtype
    eps = torch.finfo(dtype).eps

    # initialize result tensor
    cos_sim = torch.empty(
        (batch_size, filter_amount * 2, patch_size),
        device=b.device,
        dtype=dtype,
    )

    for i in range(filter_amount):
        # Extract normalized real and imaginary parts from a_normalized.
        real = a_normalized[i]
        imag = a_normalized[i + filter_amount]
        current_filters = torch.stack([real, imag])

        current_abs = abs[i].view(-1, 1)

        # Weight patches by magnitude abs.
        current_b = b * current_abs

        b_norm = torch.linalg.vector_norm(current_b, dim=-2).clamp_(min=eps)

        current_b /= b_norm.view(batch_size, 1, patch_size)

        # Calculate cosine similarity and save result
        cos_sim[:, i * 2 : i * 2 + 2, :] = current_filters @ current_b

        del current_b

    return cos_sim


def _normalize_in_place(input: torch.Tensor, dim: int):
    """
    Normalize tensor along a dimension to unit L2 norm, in-place.

    Args:
        input: Tensor to normalize. Modified in-place.
        dim: Dimension along which to compute norm and normalize.

    Notes:
        - Numerically stable: clamps norm to eps to avoid division by zero.
        - In-place operation: input is modified directly, no copy returned.
    """
    dtype = input.dtype
    eps = torch.finfo(dtype).eps

    input_norm = torch.linalg.vector_norm(input, dim=dim, keepdim=True).clamp_(min=eps)

    input /= input_norm
