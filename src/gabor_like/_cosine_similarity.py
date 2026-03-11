import torch


def _cosine_similarity(
    a_normalized: torch.Tensor, b: torch.Tensor, abs: torch.Tensor
) -> torch.Tensor:
    filter_amount = abs.size(dim=0)
    patch_size = b.size(dim=-1)
    batch_size = b.size(dim=0)

    dtype = b.dtype
    eps = torch.finfo(dtype).eps

    cos_sim = torch.empty(
        (batch_size, filter_amount * 2, patch_size),
        device=b.device,
        dtype=dtype,
    )

    for i in range(filter_amount):
        real = a_normalized[i]
        imag = a_normalized[i + filter_amount]
        current_filters = torch.stack([real, imag])

        current_abs = abs[i].view(-1, 1)

        current_b = b * current_abs

        b_norm = torch.linalg.vector_norm(current_b, dim=-2).clamp_(min=eps)

        current_b /= b_norm.view(batch_size, 1, patch_size)

        cos_sim[:, i * 2 : i * 2 + 2, :] = current_filters @ current_b

        del current_b

    return cos_sim
