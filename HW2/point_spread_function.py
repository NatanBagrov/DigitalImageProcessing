import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


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


def get_low_and_high_resolution_point_spread_function(point_spread_function, alpha, *args):
    high_resolution_kernel = point_spread_function(*args)
    # TODO: should i do it analitically or with resize?
    low_resolution_kernel = cv2.resize(high_resolution_kernel, None, fx=alpha, fy=alpha)
    # low_resolution_kernel = point_spread_function(
    #     round(args[0] * alpha) if point_spread_function == box_point_spread_function else args[0] * alpha,
    #     *args[1:])
    # TODO: should i do the same size?
    top, left = np.array(low_resolution_kernel.shape) // 2 - np.array(high_resolution_kernel.shape) // 2
    bottom, right = np.array(low_resolution_kernel.shape) // 2 + (np.array(high_resolution_kernel.shape) + 1) // 2
    low_resolution_kernel = low_resolution_kernel[top:bottom, left:right]
    assert high_resolution_kernel.shape == low_resolution_kernel.shape

    return low_resolution_kernel, high_resolution_kernel


# def apply_point_spread_function_spatial(kernel, image):
#     return convolve2d(image, kernel, mode='same')


def apply_point_spread_function_frequency(kernel, image, kernel_is_frequency=False):
    if kernel_is_frequency:
        kernel_frequency = ifftshift(kernel)
    else:
        kernel_frequency = fft2(kernel, shape=image.shape)

    image_frequency = fft2(image)
    blurred_image_frequency = kernel_frequency * image_frequency
    blurred_image = ifft2(blurred_image_frequency)
    # Why do I need fftshift here?
    blurred_image = fftshift(np.real(blurred_image))

    return blurred_image


def construct_blur_kernel_frequency(psf_low, psf_high):
    psf_low_fourier = fft2(psf_low)
    psf_high_fourier = fft2(psf_high)
    k_fourier = np.divide(
        psf_low_fourier,
        psf_high_fourier,
        out=np.zeros_like(psf_low_fourier),
        where=np.logical_not(np.isclose(psf_high_fourier, 0.0))
    )
    np.testing.assert_allclose(k_fourier, ifftshift(fftshift(k_fourier)))

    return fftshift(k_fourier)


# def construct_blur_kernel_spatial(psf_low, psf_high):
#     k_fourier = construct_blur_kernel_frequency(psf_low, psf_high)
#     k = ifft2(k_fourier)
#     k = fftshift(k)
#     k = np.real(k)
#
#     return k
