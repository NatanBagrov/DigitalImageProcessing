import os

import numpy as np
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt


def plot_images_grid(
        images,
        title=None, titles=None,
        row_headers=None, column_headers=None,
        to_frequency=False, to_spatial=False,
        disable_ticks=False,
        file_path_to_save=None
):
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

            if column == 0 and row_headers is not None:
                axes[row][column].set_ylabel(row_headers[row], rotation=90, size='small')

            current_title = ''

            if row == 0 and column_headers is not None:
                if len(current_title) > 0:
                    current_title += '\n'

                current_title += column_headers[column]

            if titles:
                if len(current_title) > 0:
                    current_title += '\n'

                current_title += titles[row][column]

            if len(current_title) > 0:
                axes[row][column].set_title(current_title, size='small')

            if disable_ticks:
                axes[row][column].tick_params(
                    axis='both',
                    which='both',
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelbottom=False,
                    labelleft=False,
                )

    if title:
        figure.suptitle(title)

    figure.tight_layout()

    if file_path_to_save is not None:
        os.makedirs(os.path.dirname(file_path_to_save), exist_ok=True)
        figure.savefig(file_path_to_save)

    plt.show()

    return figure, axes
