from external.filterbank import filterbank
import torch

def get_filterbank(N: int, sigma: float, device: str) -> torch.Tensor:
    filters, frequency_domain_sum = filterbank(N, sigma)

    filters = filters.astype('complex64')
    frequency_domain_sum = frequency_domain_sum.astype('float32')

    frequency_domain_sum = torch.tensor(frequency_domain_sum)
    filter = torch.tensor(filters[1].real)
    filter_norm = torch.linalg.vector_norm(filter)
    filter_normalized = filter / filter_norm
    return filter_normalized, frequency_domain_sum