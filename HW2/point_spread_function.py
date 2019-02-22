from itertools import islice

import numpy as np
from scipy.signal import convolve2d
from scipy.linalg import circulant
from scipy import sparse
from tqdm import tqdm

from utils import conjugate_transpose


def gaussian_point_spread_function(sigma, height, width):
    y = np.arange(-height // 2, (height + 1) // 2)
    x = np.arange(-width // 2, (width + 1) // 2)
    xx, yy = np.meshgrid(x, y)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))

    return kernel / np.sum(kernel)


def box_point_spread_function(size, height, width):
    kernel = np.zeros((height, width), dtype=np.float)
    left = width // 2 - size // 2
    right = width // 2 + (size + 1) // 2
    top = height // 2 - size // 2
    bottom = height // 2 + (size + 1) // 2
    kernel[top:bottom, left:right] = 1.0
    assert np.all(np.isin(np.sum(kernel, axis=1), (0, size)))
    assert np.all(np.isin(np.sum(kernel, axis=0), (0, size)))

    return kernel / np.sum(kernel)


def horizontal_derivative_kernel():
    return [[1, -1]]


def vertical_derivative_kernel():
    return [[1, ], [-1, ]]


def get_low_and_high_resolution_point_spread_function(
        point_spread_function,
        alpha,
        high_size_argument,
        high_height, high_width):
    high_resolution_kernel = point_spread_function(high_size_argument, high_height, high_width)
    low_size_argument = high_size_argument * alpha
    low_height = round(alpha * high_height)
    low_width = round(alpha * high_width)

    if point_spread_function == box_point_spread_function:
        low_size_argument = round(low_size_argument)

    low_resolution_kernel = point_spread_function(low_size_argument, low_height, low_width)

    return low_resolution_kernel, high_resolution_kernel


def apply_point_spread_function_spatial(kernel, image):
    # TODO: putting mode='full' here reduces artifact in Wiener
    return convolve2d(image, kernel, mode='same', boundary='wrap')


def kernel_to_doubly_block_circulant(kernel, second_shape, dense=False):
    kernel = np.array(kernel)
    second_shape = np.array(second_shape)
    result_shape = np.array(second_shape) + np.array(kernel.shape) - 1
    kernel = np.pad(
        kernel,
        (
            (0, result_shape[0] - kernel.shape[0]),
            (0, result_shape[1] - kernel.shape[1]),
        ),
        mode='constant',
        constant_values=0.0,
    )

    # TODO: vectorize code below
    blocks = map(
        lambda row: circulant(row)[:, :second_shape[1]],
        kernel
    )

    if not dense:
        blocks = map(sparse.dok_matrix, blocks)

    blocks = list(islice(blocks, None, int(result_shape[0]), None))
    doubly_block_circulant = (np.zeros if dense else sparse.dok_matrix)((
        np.prod(result_shape),
        np.prod(second_shape),
    ))
    status_bar = tqdm(
        total=result_shape[0] * second_shape[0],
        desc='Doubly block circulant matrix construction'
    )

    for block_row in range(result_shape[0]):
        row_begin = block_row * result_shape[1]
        row_end = row_begin + result_shape[1]

        for block_column in range(second_shape[0]):
            column_begin = block_column * second_shape[1]
            column_end = column_begin + second_shape[1]
            doubly_block_circulant[row_begin:row_end, column_begin:column_end] =\
                blocks[(block_row - block_column + result_shape[0]) % result_shape[0]]
            status_bar.update(1)

    status_bar.close()
    return doubly_block_circulant


def construct_blur_kernel_spatial(psf_low, psf_high):
    blur_kernel_shape = np.array(psf_low.shape) - np.array(psf_high.shape) + 1
    psf_high_doubly_block_circulant = kernel_to_doubly_block_circulant(psf_high, blur_kernel_shape, dense=True)
    blur_kernel = \
        np.linalg.inv(
            conjugate_transpose(psf_high_doubly_block_circulant)
            @ psf_high_doubly_block_circulant) \
        @ conjugate_transpose(psf_high_doubly_block_circulant) \
        @ psf_low.flatten()

    return blur_kernel.reshape(blur_kernel_shape)
