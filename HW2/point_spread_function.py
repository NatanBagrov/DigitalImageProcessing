from itertools import islice

import numpy as np
from scipy.signal import convolve2d
from scipy.linalg import circulant


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
    return [[-1, 0, +1]]


def vertical_derivative_kernel():
    return [[-1], [0], [+1]]


def gradient_doubly_block_circulant(image_shape):
    horizontal_dbc = kernel_to_doubly_block_circulant(horizontal_derivative_kernel(), image_shape)
    vertical_dbc = kernel_to_doubly_block_circulant(vertical_derivative_kernel(), image_shape)
    dbc = np.hstack((
        horizontal_dbc,
        vertical_dbc
    ))

    return dbc


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
    return convolve2d(image, kernel, mode='full')

# def apply_point_spread_function_frequency(kernel, image, kernel_is_frequency=False):
#     if kernel_is_frequency:
#         kernel_frequency = ifftshift(kernel)
#     else:
#         kernel_frequency = fft2(kernel, shape=image.shape)
#
#     image_frequency = fft2(image)
#     blurred_image_frequency = kernel_frequency * image_frequency
#     blurred_image = ifft2(blurred_image_frequency)
#     # Why do I need fftshift here?
#     blurred_image = fftshift(np.real(blurred_image))
#
#     return blurred_image


# def construct_blur_kernel_frequency(psf_low, psf_high):
#     psf_low_fourier = fft2(psf_low)
#     psf_high_fourier = fft2(psf_high)
#     k_fourier = np.divide(
#         psf_low_fourier,
#         psf_high_fourier,
#         out=np.zeros_like(psf_low_fourier),
#         where=np.logical_not(np.isclose(psf_high_fourier, 0.0))
#     )
#     np.testing.assert_allclose(k_fourier, ifftshift(fftshift(k_fourier)))
#
#     return fftshift(k_fourier)


# TODO: Sparsify it
def kernel_to_doubly_block_circulant(kernel, second_shape):
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
    blocks = list(islice(map(
        lambda row: circulant(row)[:, :second_shape[1]],
        kernel
    ), None, int(result_shape[0]), None))
    doubly_block_circulant = np.zeros((
        np.prod(result_shape),
        np.prod(second_shape),
    ))



    for block_row in range(result_shape[0]):
        row_begin = block_row * result_shape[1]
        row_end = row_begin + result_shape[1]

        for block_column in range(second_shape[0]):
            column_begin = block_column * second_shape[1]
            column_end = column_begin + second_shape[1]
            doubly_block_circulant[row_begin:row_end, column_begin:column_end] =\
                blocks[(block_row - block_column + result_shape[0]) % result_shape[0]]

    # # TODO: BY DEFINITION - VECTORIZE IT
    # for result_row in range(result_shape[0]):
    #     for result_column in range(result_shape[1]):
    #         row = result_row * result_shape[1] + result_column
    #
    #         for kernel_row in range(kernel.shape[0]):
    #             for kernel_column in range(kernel.shape[1]):
    #                 second_row = result_row - kernel_row
    #                 second_column = result_column - kernel_column
    #
    #                 if np.all(0 <= np.array([second_row, second_column])) \
    #                         and np.all(np.array([second_row, second_column]) < second_shape):
    #                     column = second_row * second_shape[0] + second_column
    #                     doubly_block_circulant[row][column] = kernel[kernel_row][kernel_column]

    return doubly_block_circulant


def construct_blur_kernel_spatial(psf_low, psf_high):
    blur_kernel_shape = np.array(psf_low.shape) - np.array(psf_high.shape) + 1
    psf_high_doubly_block_circulant = kernel_to_doubly_block_circulant(psf_high, blur_kernel_shape)
    # TODO: put analytical solution here
    blur_kernel, residuals, rank, s = np.linalg.lstsq(psf_high_doubly_block_circulant, psf_low.flatten(), rcond=-1)

    return blur_kernel.reshape(blur_kernel_shape)


# def construct_blur_kernel_spatial(psf_low, psf_high):
#     k_fourier = construct_blur_kernel_frequency(psf_low, psf_high)
#     k = ifft2(k_fourier)
#     k = fftshift(k)
#     k = np.real(k)
#
#     return k
