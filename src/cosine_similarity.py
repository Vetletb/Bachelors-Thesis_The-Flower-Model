import torch

def cosine_similarity(a_normalized: torch.Tensor, b:torch.Tensor, abs:torch.Tensor) -> torch.Tensor:

    abs_flatten = abs.view(-1, 1)
    a_normalized_flatten = a_normalized.view(-1, 1)
    print(a_normalized_flatten.shape)
    b_new = b * abs_flatten
    print(b.shape)
    b_new_norm = torch.linalg.vector_norm(b_new, dim=-2)

    return torch.linalg.vecdot(a_normalized_flatten, b_new, dim=-2) / b_new_norm
