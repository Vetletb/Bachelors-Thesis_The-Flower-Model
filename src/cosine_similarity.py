import torch


def cosine_similarity(
    a_normalized: torch.Tensor, b: torch.Tensor, abs: torch.Tensor
) -> torch.Tensor:
    abs_flatten = abs.view(-1, 1)

    b *= abs_flatten
    b_norm = torch.linalg.vector_norm(b, dim=-2)

    patch_size = b.size(dim=-1)
    batch_size = b.size(dim=0)

    return torch.matmul(a_normalized, b) / b_norm.view(
        batch_size, 1, patch_size
    )
