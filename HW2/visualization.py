import os

import numpy as np
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt


def plot_images_grid(images, title=None, titles=None, to_frequency=False, to_spatial=False, file_path_to_save=None):
    assert not to_frequency or not to_spatial

    rows = len(images)
    columns = max(map(len, images))

    figure, axes = plt.subplots(rows, columns, squeeze=False)

    for row in range(rows):
        for column in range(len(images[row])):
            if to_frequency:
                image = np.log(np.abs(fftshift(fft2(images[row][column]))))
            elif to_spatial:
                image = ifft2(ifftshift(images[row][column]))
            else:
                image = images[row][column]
                extent = None

            extent = [
                -image.shape[1] // 2,
                (image.shape[1] + 1) // 2,
                -image.shape[0] // 2,
                (image.shape[0] + 1) // 2,
            ]

            axes[row][column].imshow(image, cmap='gray', extent=extent)

            if titles:
                axes[row][column].set_title(titles[row][column])
    if title:
        figure.suptitle(title)

    if file_path_to_save is not None:
        os.makedirs(os.path.dirname(file_path_to_save), exist_ok=True)
        figure.savefig(file_path_to_save)

    plt.show()

    return figure, axes
