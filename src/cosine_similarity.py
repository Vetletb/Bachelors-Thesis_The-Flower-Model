import torch

def cosine_similarity(a_normalized: torch.Tensor, b:torch.Tensor, abs:torch.Tensor) -> torch.Tensor:
    abs_flatten = abs.view(-1, 1)
    
    b_new = b * abs_flatten
    b_new_norm = torch.linalg.vector_norm(b_new, dim=-2)

    patch_size = b_new.size(dim=-1)
    batch_size = b_new.size(dim=0)
    
    return torch.matmul(a_normalized, b_new) / b_new_norm.view(batch_size, 1, patch_size)
