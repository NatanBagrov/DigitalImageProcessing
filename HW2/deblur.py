import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.signal import convolve2d, correlate2d

from point_spread_function import \
    gradient_doubly_block_circulant, \
    kernel_to_doubly_block_circulant, \
    horizontal_derivative_kernel, \
    vertical_derivative_kernel
from optimization import \
    alternating_direction_method_of_multipliers, \
    de_degradation_f_step_multiplication, \
    de_degradation_f_step_convolution


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


def inverse_filter_1(blurred_image, kernel, desired_shape=None):
    if desired_shape is None:
        desired_shape = blurred_image.shape

    # TODO: can I apply Wienner filter in frequency domain
    kernel_frequency = fft2(kernel, shape=desired_shape)
    blurred_image_frequency = fft2(blurred_image, shape=desired_shape)
    de_blurred_image_frequency = np.divide(
        blurred_image_frequency,
        kernel_frequency,
        out=np.zeros_like(kernel_frequency),
        where=np.logical_not(np.isclose(kernel_frequency, 0.0)),
    )
    de_blurred_image = ifft2(de_blurred_image_frequency, shape=desired_shape).real

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
    return convolve2d(low_resolution_image, kernel, mode='same')


def total_variation_de_blurring_spatial(low_resolution_image, kernel, lambda_=1.0, rho=1.0, epsilon=1e-2, callback=lambda _: False):
    def call_back_wrapper(image):
        gradient_high_resolution_image = gradient @ image
        gradient_horizontal_high_resolution_image = \
            gradient_high_resolution_image[:gradient_high_resolution_image.size // 2]
        gradient_vertical_high_resolution_image = \
            gradient_high_resolution_image[gradient_high_resolution_image.size // 2:]
        gradient_abs_high_resolution_image = \
            np.sqrt(gradient_horizontal_high_resolution_image ** 2 + gradient_vertical_high_resolution_image ** 2)
        low_resolution_image_estimate = kernel @ image
        image = np.reshape(image, high_resolution_image_shape)
        low_resolution_image_estimate = low_resolution_image_estimate.reshape(low_resolution_image.shape)
        l2_loss = np.linalg.norm(low_resolution_image - low_resolution_image_estimate, ord=2)
        print('||l-k*h||_2={}'.format(l2_loss))
        tv_loss = np.linalg.norm(gradient_abs_high_resolution_image, ord=1)
        print('||h||_TV={}'.format(tv_loss))
        loss = l2_loss + lambda_ * tv_loss
        print('||l-k*h||_2+lambda*||h||_TV={}'.format(loss))

        return callback(image)

    high_resolution_image_shape = np.array(low_resolution_image.shape) - np.array(kernel.shape) + 1
    gradient = gradient_doubly_block_circulant(high_resolution_image_shape)
    kernel = kernel_to_doubly_block_circulant(kernel, high_resolution_image_shape)
    f_step = de_degradation_f_step_multiplication(low_resolution_image.flatten(), kernel, gradient, p=rho)
    high_resolution_image = np.reshape(alternating_direction_method_of_multipliers(
        gradient,
        high_resolution_image_shape,
        f_step,
        l=lambda_,
        p=rho,
        epsilon=epsilon,
        callback=call_back_wrapper,
    ), high_resolution_image_shape)
    assert all(high_resolution_image.shape == high_resolution_image_shape)

    return high_resolution_image


def total_variation_de_blurring_frequency(low_resolution_image, kernel, lambda_=1.0, rho=1.0, epsilon=1e-2, callback=lambda _: False):
    hdk = horizontal_derivative_kernel()
    vdk = vertical_derivative_kernel()

    def gradient(image):
        return [
            convolve2d(image, hdk, mode='same'),
            convolve2d(image, vdk, mode='same'),
        ]

    f_step = de_degradation_f_step_convolution(low_resolution_image, kernel, [hdk, vdk], p=rho)

    high_resolution_image = alternating_direction_method_of_multipliers(
        gradient,
        low_resolution_image,
        f_step,
        l=lambda_,
        p=rho,
        epsilon=epsilon,
        callback=callback,
    )

    return high_resolution_image


def correlate(f, h):
    return correlate2d(f, h, mode='same', boundary='wrap')


def convolve(f, h):
    return convolve2d(f, h, mode='same', boundary='wrap')


dx = [[1, ], [-1, ]]
dy = [[1, -1]]


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
        np.square(np.abs(fft2(dx, shape=f.shape))) + \
        np.square(np.abs(fft2(dy, shape=f.shape)))
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
