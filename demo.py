import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gaussian import gaussian

matplotlib.rcParams.update({"figure.figsize": np.array(matplotlib.rcParams["figure.figsize"]) * [2., 1.]})

np.set_printoptions(precision=2)

N = 101
n1 = np.arange(N).reshape((-1, 1)).repeat(N, axis=1) - (N - 1) / 2
n2 = n1.T
n = np.array([n1, n2])

sigma = 100.

fig, axes = plt.subplots(1, 4)
images = []

extent = [-(N - 1) / 2, (N - 1) / 2, (N - 1) / 2, -(N - 1) / 2]
images.append(axes[0].imshow(gaussian(n, np.array([0., 0.]), sigma, False), extent=extent))
axes[0].set_title('Fourier domain')
frequency_domain = gaussian(n, np.array([0., 0.]), sigma, True)
images.append(axes[1].imshow(frequency_domain, extent=extent))
axes[1].set_title('Fourier domain')
spatial_domain = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(frequency_domain)))
images.append(axes[2].imshow(spatial_domain.real,
                             vmin=np.min([spatial_domain.real, spatial_domain.imag]),
                             vmax=np.max([spatial_domain.real, spatial_domain.imag]),
                             extent=extent))
axes[2].set_title('Spatial domain,\nreal parts')
images.append(axes[3].imshow(spatial_domain.imag,
                             vmin=np.min([spatial_domain.real, spatial_domain.imag]),
                             vmax=np.max([spatial_domain.real, spatial_domain.imag]),
                             extent=extent))
axes[3].set_title('Spatial domain,\nimaginary parts')


def onclick(event):
    if event.button == 1 and (event.inaxes == axes[0] or event.inaxes == axes[1]):
        frequency_domain = gaussian(n, np.array([event.ydata, event.xdata]), sigma, False)
        images[0].set_data(frequency_domain)
        images[0].set_clim(np.min(frequency_domain), np.max(frequency_domain))
        frequency_domain = gaussian(n, np.array([event.ydata, event.xdata]), sigma, True)
        images[1].set_data(frequency_domain)
        images[1].set_clim(np.min(frequency_domain), np.max(frequency_domain))
        spatial_domain = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(frequency_domain)))
        images[2].set_data(spatial_domain.real)
        images[2].set_clim(np.min([spatial_domain.real, spatial_domain.imag]), np.max([spatial_domain.real, spatial_domain.imag]))
        images[3].set_data(spatial_domain.imag)
        images[3].set_clim(np.min([spatial_domain.real, spatial_domain.imag]), np.max([spatial_domain.real, spatial_domain.imag]))

        fig.canvas.draw()
        fig.canvas.flush_events()


fig.canvas.mpl_connect('motion_notify_event', onclick)
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
