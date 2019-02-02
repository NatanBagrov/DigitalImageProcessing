import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.signal import convolve2d

from point_spread_function import gradient_doubly_block_circulant, kernel_to_doubly_block_circulant
from optimization import alternating_direction_method_of_multipliers, de_degradation_f_step


def get_inverse_filter(kernel):
    kernel_frequency = fft2(kernel)
    filter_frequency = np.divide(
        1.0,
        kernel_frequency,
        out=np.zeros_like(kernel_frequency),
        where=np.logical_not(np.isclose(kernel_frequency, 0.0)),
    )
    filter_spatial = ifft2(filter_frequency).real

    return filter_spatial


def inverse_filter_1(blurred_image, kernel):
    # TODO: can I apply Wienner filter in frequency domain
    kernel_frequency = fft2(kernel, shape=blurred_image.shape)
    blurred_image_frequency = fft2(blurred_image)
    de_blurred_image_frequency = np.divide(
        blurred_image_frequency,
        kernel_frequency,
        out=np.zeros_like(kernel_frequency),
        where=np.logical_not(np.isclose(kernel_frequency, 0.0)),
    )
    de_blurred_image = ifft2(de_blurred_image_frequency).real

    # filter_spatial = kernel_to_doubly_block_circulant(kernel, blurred_image.shape)
    # de_blurred_image = estimate_high_resolution_image(blurred_image, filter_spatial)

    return de_blurred_image


# def inverse_filter_frequency(blurred_image, kernel_frequency):
#     top = blurred_image.shape[0] // 2 - kernel_frequency.shape[0] // 2
#     bottom = (blurred_image.shape[0] + 1) // 2 - (kernel_frequency.shape[0] + 1) // 2
#     left = blurred_image.shape[1] // 2 - kernel_frequency.shape[1] // 2
#     right = (blurred_image.shape[1] + 1) // 2 - (kernel_frequency.shape[1] + 1) // 2
#
#     kernel_frequency = np.pad(
#         kernel_frequency,
#         ((top, bottom), (left, right)),
#         mode='constant',
#         constant_values=0.0,
#     )
#
#     assert kernel_frequency.shape == blurred_image.shape
#
#     blurred_image_frequency = fftshift(fft2(blurred_image))
#     de_blurred_image_frequency = np.divide(
#         blurred_image_frequency,
#         kernel_frequency,
#         out=np.zeros_like(blurred_image_frequency),
#         where=np.logical_not(np.isclose(kernel_frequency, 0.0)),
#     )
#     de_blurred_image = ifft2(ifftshift(de_blurred_image_frequency))
#     de_blurred_image = de_blurred_image.real
#
#     return de_blurred_image


def bilinear_kernel():
    # TODO: is it the right kernel
    kernel = np.array([
        [0.0, 1.0, 0.0, ],
        [1.0, 0.0, 1.0, ],
        [0.0, 1.0, 0.0, ],
    ])

    return kernel


def bicubic_kernel():
    a = -0.75
    x = np.arange(-1, 2)
    y = np.arange(-1, 2)
    x, y = np.meshgrid(x, y)
    absx = np.sqrt(x ** 2 + y ** 2)
    kernel = \
        (absx <= 1.0) * ((a + 2.0) * (absx ** 3) - (a + 3) * (absx ** 2) + 1.0) + \
        ((1.0 < absx) & (absx < 2.0)) * (a * (absx ** 3) - 5.0 * a * (absx ** 2) + 8.0 * a * absx - 4.0 * a)

    return kernel


def estimate_high_resolution_image(low_resolution_image, kernel):
    return convolve2d(low_resolution_image, kernel, mode='same')


def total_variation_de_blurring(low_resolution_image, kernel, rho=1.0, calback=lambda _: False):
    high_resolution_image_shape = np.array(low_resolution_image.shape) - np.array(kernel.shape)
    gradient = gradient_doubly_block_circulant(high_resolution_image_shape)
    kernel = kernel_to_doubly_block_circulant(kernel, high_resolution_image_shape)
    f_step = de_degradation_f_step(low_resolution_image, kernel, gradient, p=rho)
    high_resolution_image = np.reshape(alternating_direction_method_of_multipliers(
        gradient,
        f_step,
        p=rho,
        callback=calback,
    ), high_resolution_image_shape)
    assert high_resolution_image.shape == high_resolution_image_shape

    return high_resolution_image
