import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
from external.gaussian import gaussian

def filterbank(N: int, sigma: float) -> np.ndarray:

    # scale = 0.102

    # matplotlib.rcParams.update({
    #     "figure.figsize": np.array([6.4, 6.4]) * scale,
    #     'lines.markersize': matplotlib.rcParams["lines.markersize"] * scale * 2.,
    #     "font.family": "serif",
    #     "text.usetex": True,
    #     "pgf.rcfonts": False,
    #     "pgf.preamble": r"\usepackage{amsmath}"
    # })

    # np.set_printoptions(precision=2)

    n1 = np.arange(N).reshape((-1, 1)).repeat(N, axis=1) - (N - 1) / 2
    n2 = n1.T
    n = np.array([n1, n2])

    # mu = np.array([20., 20.])

    # plt.xticks(())
    # plt.yticks(())
    # plt.tight_layout(pad=0.0)

    dr = 6.
    dtheta = np.pi / 22.

    frequency_domain = gaussian(n, np.array([0., 0.]), sigma, True)
    frequency_domain_sum = frequency_domain

    center_filter = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(frequency_domain)))

    # plt.imshow(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(frequency_domain))).real)
    # plt.savefig(f'figure2-{0.:.2f}.pdf', bbox_inches='tight', pad_inches=.0)

    filter_list = []

    for theta in np.arange(0., np.pi / 2. + dtheta, dtheta):
        for r in np.arange(dr, dr * 7. + dr, dr):
            filter = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gaussian(n, r * np.array([np.cos(theta), np.sin(theta)]), sigma, True))))

            # plt.imshow(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gaussian(n, r * np.array([np.cos(theta), np.sin(theta)]), sigma, True)))).real)
            # plt.savefig(f'figure2-{r:.2f}-{theta:.2f}.pdf', bbox_inches='tight', pad_inches=.0)

            filter_list.append(filter)

    filters = np.stack(filter_list, axis=0)

    for theta in np.arange(0., 2. * np.pi, dtheta):
        for r in np.arange(dr, dr * 8. + dr, dr):
            frequency_domain_sum += gaussian(n, r * np.array([np.cos(theta), np.sin(theta)]), sigma, True)

    # plt.imshow(frequency_domain_sum, vmin=0.)
    # plt.savefig(f'figure2-sum.pdf', bbox_inches='tight', pad_inches=.0)

    return center_filter, filters, frequency_domain_sum