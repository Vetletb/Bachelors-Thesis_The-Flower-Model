import torch

def cosine_similarity(a_normalized: torch.Tensor, b:torch.Tensor, abs:torch.Tensor) -> torch.Tensor:
    
    b_new = torch.mul(b, abs)
    b_new_norm = torch.linalg.vector_norm(b_new)

    return torch.linalg.vecdot(a_normalized, b_new) / b_new_norm
