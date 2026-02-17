import torch
from torch.nn.functional import normalize


def _cosine_similarity(
    a_normalized: torch.Tensor, b: torch.Tensor, abs: torch.Tensor
) -> torch.Tensor:
    abs_flatten = abs.view(-1, 1)

    b *= abs_flatten
    b_normalized = normalize(b, dim=-2)

    return a_normalized @ b_normalized
