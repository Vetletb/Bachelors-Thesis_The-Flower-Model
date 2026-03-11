import numpy as np
from gabor_like.external._gaussian import _gaussian


def _gabor_like_filters(N: int, sigma: float) -> np.ndarray:
    n1 = np.arange(N).reshape((-1, 1)).repeat(N, axis=1) - (N - 1) / 2
    n2 = n1.T
    n = np.array([n1, n2])

    radius = int((N - 1) / 2 * 0.7)
    dr = radius / 10
    radius = int((N - 1) / 2)
    dr = radius / 20
    steps = radius / dr
    dtheta = np.arccos(1 - (dr**2 / (2 * radius**2)))
    
    dr *= 0.5


    frequency_domain = _gaussian(n, np.array([0.0, 0.0]), sigma, True)

    filter_list = []

    filter_list.append(
        np.stack(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(frequency_domain))))
    )

    for theta in np.arange(3 * np.pi / 2, 5 * np.pi / 2 + dtheta, dtheta):
        for r in np.arange(dr, dr * steps + dr, dr):
            filter = np.fft.fftshift(
                np.fft.ifft2(
                    np.fft.ifftshift(
                        _gaussian(
                            n, r * np.array([np.cos(theta), np.sin(theta)]), sigma, True
                        )
                    )
                )
            )
            filter_list.append(filter)
    filters = np.stack(filter_list, axis=0)

    return filters
