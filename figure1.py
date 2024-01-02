import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gaussian import gaussian

scale = 0.24

matplotlib.rcParams.update({
    "figure.figsize": np.array([6.4, 6.4]) * scale,
    'lines.markersize': matplotlib.rcParams["lines.markersize"] * scale * 2.,
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{amsmath}"
})

np.set_printoptions(precision=2)

N = 101
n1 = np.arange(N).reshape((-1, 1)).repeat(N, axis=1) - (N - 1) / 2
n2 = n1.T
n = np.array([n1, n2])

mu = np.array([20., 20.])
sigma = 100.

plt.xticks(())
plt.yticks(())
plt.tight_layout(pad=0.0)

plt.imshow(gaussian(n, mu, sigma, False))
plt.savefig('figure1a.pdf', bbox_inches='tight', pad_inches=.0)

frequency_domain = gaussian(n, mu, sigma, True)

plt.imshow(frequency_domain)
plt.savefig('figure1b.pdf', bbox_inches='tight', pad_inches=.0)

spatial_domain = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(frequency_domain)))

plt.imshow(spatial_domain.real, vmin=np.min([spatial_domain.real, spatial_domain.imag]), vmax=np.max([spatial_domain.real, spatial_domain.imag]))
plt.savefig('figure1c.pdf', bbox_inches='tight', pad_inches=.0)

plt.imshow(spatial_domain.imag, vmin=np.min([spatial_domain.real, spatial_domain.imag]), vmax=np.max([spatial_domain.real, spatial_domain.imag]))
plt.savefig('figure1d.pdf', bbox_inches='tight', pad_inches=.0)

print(f"Sum image.real: {np.sum(spatial_domain.real)}")
print(f"Sum image.imag: {np.sum(spatial_domain.imag)}")
print(f"Cosine similarity: {np.sum(spatial_domain.real * spatial_domain.imag)/(np.linalg.norm(spatial_domain.real) * np.linalg.norm(spatial_domain.imag))}")
