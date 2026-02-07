from external.filterbank import filterbank
import torch


def create_filterbank(N: int, sigma: float, device: str) -> torch.Tensor:
    filters, frequency_domain_sum = filterbank(N, sigma)

    filters = filters.astype("float32")
    frequency_domain_sum = frequency_domain_sum.astype("float32")

    frequency_domain_sum = torch.tensor(frequency_domain_sum, device=device)
    filters = torch.tensor(filters, device=device)

    filter_amount = filters.size(dim=0)
    filters = filters.view(filter_amount, -1)

    filters_norm = torch.linalg.vector_norm(filters, dim=-1)
    filters_normalized = filters / filters_norm.view(-1, 1)
    return filters_normalized, frequency_domain_sum
