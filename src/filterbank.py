from external.filterbank import filterbank
import torch

def get_filterbank(N: int, sigma: float) -> torch.Tensor:
    arr1, arr2, frequency_domain_sum = filterbank(N, sigma)
    arr1 = arr1.astype('complex64')
    arr2 = arr2.astype('complex64')
    frequency_domain_sum = frequency_domain_sum.astype('float32')
    frequency_domain_sum = torch.tensor(frequency_domain_sum)
    filter = torch.tensor(arr2[0].real)
    filter_norm = torch.linalg.vector_norm(filter)
    filter_normalized = filter / filter_norm
    return filter_normalized, frequency_domain_sum