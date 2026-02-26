import torch


def _cosine_similarity(
    a_normalized: torch.Tensor, b: torch.Tensor, abs: torch.Tensor
) -> torch.Tensor:
    cos_sim_list = []

    filter_amount = abs.size(dim=0)
    for i in range(filter_amount):
        real = a_normalized[i]
        imag = a_normalized[i + filter_amount]
        current_filters = torch.stack([real, imag, -real, -imag])

        current_abs = abs[i].view(-1, 1)

        current_b = b * current_abs

        dtype = current_b.dtype
        eps = torch.finfo(dtype).eps
        b_norm = torch.linalg.vector_norm(current_b, dim=-2).clamp_(min=eps)

        patch_size = current_b.size(dim=-1)
        batch_size = current_b.size(dim=0)
        current_b /= b_norm.view(batch_size, 1, patch_size)

        cos_sim = current_filters @ current_b
        cos_sim_list.append(cos_sim)

    cos_sim = torch.hstack(cos_sim_list)

    return cos_sim
