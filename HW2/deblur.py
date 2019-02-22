import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.signal import convolve2d

from utils import correlate, convolve
from point_spread_function import horizontal_derivative_kernel, vertical_derivative_kernel


def inverse_filter_1(blurred_image, kernel, desired_shape=None):
    if desired_shape is None:
        desired_shape = blurred_image.shape

    # TODO: can I apply Wiener filter in frequency domain
    kernel_frequency = fft2(kernel, shape=desired_shape)
    blurred_image_frequency = fft2(blurred_image, shape=desired_shape)
    de_blurred_image_frequency = np.divide(
        blurred_image_frequency,
        kernel_frequency,
        out=np.zeros_like(kernel_frequency),
        where=np.logical_not(np.isclose(kernel_frequency, 0.0)),
    )
    de_blurred_image = np.real(ifft2(de_blurred_image_frequency, shape=desired_shape))
    # TODO: I will be happy to know exactly is how and why it works (for mode=same)
    kernel_height, kernel_width = kernel.shape
    top_left     = de_blurred_image[:-kernel_height // 2, :-kernel_width // 2]
    bottom_left  = de_blurred_image[-kernel_height // 2:, :-kernel_width // 2]
    top_right    = de_blurred_image[:-kernel_height // 2, -kernel_width // 2:]
    bottom_right = de_blurred_image[-kernel_height // 2:, -kernel_width // 2:]
    de_blurred_image = np.block([
        [bottom_right, bottom_left],
        [top_right, top_left],
    ])

    # filter_spatial = kernel_to_doubly_block_circulant(kernel, blurred_image.shape)
    # de_blurred_image = estimate_high_resolution_image(blurred_image, filter_spatial)

    return de_blurred_image


# TODO: are these the right kernels?!!
def bilinear_kernel():
    kernel = np.array([
        [0.0, 1.0, 0.0, ],
        [1.0, 0.0, 1.0, ],
        [0.0, 1.0, 0.0, ],
    ]) / 4.0

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

    kernel /= kernel.sum()

    return kernel


def estimate_high_resolution_image(low_resolution_image, kernel):
    return convolve(low_resolution_image, kernel)


# DEBUG THIS FUNCTION!!!
# def total_variation_de_blurring_frequency(low_resolution_image, kernel, lambda_=1.0, rho=1.0, epsilon=1e-2, callback=lambda _: False):
#     hdk = horizontal_derivative_kernel()
#     vdk = vertical_derivative_kernel()
#
#     def gradient(image):
#         return [
#             convolve2d(image, hdk, mode='same', boundary='wrap'),
#             convolve2d(image, vdk, mode='same', boundary='wrap'),
#         ]
#
#     f_step = de_degradation_f_step_convolution(low_resolution_image, kernel, [hdk, vdk], p=rho)
#
#     high_resolution_image = alternating_direction_method_of_multipliers(
#         gradient,
#         low_resolution_image,
#         f_step,
#         l=lambda_,
#         p=rho,
#         epsilon=epsilon,
#         callback=callback,
#     )
#
#     return high_resolution_image

def d(u):
    # return convolve(u, dx), convolve(u, dy)
    return \
        np.hstack((np.diff(u, 1, 1), np.reshape(u[:, 0] - u[:, -1], (-1, 1)))), \
        np.vstack((np.diff(u, 1, 0), np.reshape(u[0, :] - u[-1, :], (1, -1))))


def dt(x, y):
    # return correlate(x, dx) + correlate(y, dy)
    return np.hstack((
        np.reshape(x[:, -1] - x[:, 0], (-1, 1)),
        -np.diff(x, 1, 1),
    )) + np.vstack((
        np.reshape(y[-1, :] - y[0, :], (1, -1)),
        -np.diff(y, 1, 0),
    ))


def total_variation_de_blurring(f, h, mu, beta, gamma, epsilon, callback=lambda _: False):

    ktf = correlate(f, h)
    eigsdtd = \
        np.square(np.abs(fft2(horizontal_derivative_kernel(), shape=f.shape))) + \
        np.square(np.abs(fft2(vertical_derivative_kernel(), shape=f.shape)))
    assert not np.any(np.isnan(eigsdtd))
    assert not np.any(np.isinf(eigsdtd))
    eigsktk = np.square(np.abs(fft2(h, shape=f.shape)))
    assert not np.any(np.isnan(eigsktk))
    assert not np.any(np.isinf(eigsktk))

    x = f
    lambda1 = np.zeros(f.shape)
    lambda2 = np.zeros(f.shape)
    change = np.inf
    d1, d2 = d(x)

    while change > epsilon:
        assert not np.any(np.isnan(d1))
        assert not np.any(np.isinf(d2))

        z1 = d1 + lambda1 / beta
        assert not np.any(np.isnan(z1))
        assert not np.any(np.isinf(z1))
        z2 = d2 + lambda2 / beta
        assert not np.any(np.isnan(z2))
        assert not np.any(np.isinf(z2))
        v = np.sqrt(np.square(z1) + np.square(z2))
        assert not np.any(np.isnan(v))
        assert not np.any(np.isinf(v))
        v[0.0 == v] = 1.0
        v = np.maximum(v - 1.0 / beta, 0.0) / v
        assert not np.any(np.isnan(v))
        assert not np.any(np.isinf(v))
        y1 = z1 * v
        assert not np.any(np.isnan(y1))
        assert not np.any(np.isinf(y1))
        y2 = z2 * v
        assert not np.any(np.isnan(y2))
        assert not np.any(np.isinf(y2))

        xp = x
        x = xp
        x = (mu * ktf - dt(lambda1, lambda2)) / beta + dt(y1, y2)
        assert not np.any(np.isnan(x))
        assert not np.any(np.isinf(x))
        # x[np.abs(x) > np.mean(np.abs(x)) + np.std(np.abs(x))] = 0.0
        x = fft2(x) / (eigsdtd + (mu / beta) * eigsktk)
        assert not np.any(np.isnan(x))
        assert not np.any(np.isinf(x))
        x = np.real(ifft2(x))
        assert not np.any(np.isnan(x))
        assert not np.any(np.isinf(x))
        # import matplotlib.pyplot as plt; plt.imshow(x, cmap='gray'); plt.colorbar(); plt.show()
        change = np.linalg.norm(x - xp) / np.linalg.norm(x)

        d1, d2 = d(x)
        lambda1 -= gamma * beta * (y1 - d1)
        assert not np.any(np.isnan(lambda1))
        assert not np.any(np.isinf(lambda1))
        lambda2 -= gamma * beta * (y2 - d2)
        assert not np.any(np.isnan(lambda2))
        assert not np.any(np.isinf(lambda2))

        if callback(x):
            break

    return x
