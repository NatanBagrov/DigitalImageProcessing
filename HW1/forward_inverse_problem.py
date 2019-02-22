import cv2

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy import fftpack

from psf import generate_psfs, psf_fine
from readers import read_original_image


def forward_blur(original_image, psfs):
    print("Generating {} blurred images (from PSFs)".format(len(psfs)))
    # return [cv2.filter2D(original_image.astype(float), -1, psf) for psf in psfs]
    return [sig.convolve2d(original_image.astype(float), psf, mode='same', boundary='wrap') for psf in psfs]


def inverse_deblur(blurred_images, fft_estimator):
    print("Generating a de-blurred image using {} blurred images".format(len(blurred_images)))
    blurred_ffts = fftpack.fft2(blurred_images)
    estimated_fft = fft_estimator(blurred_ffts)
    result = fftpack.ifft2(estimated_fft)
    result = np.absolute(result)
    return result


def estimate_fft_using_max(blurred_ffts):
    dim = blurred_ffts[0].shape[0]
    abs_ffts = np.abs(blurred_ffts)
    argmax_ffts = np.argmax(abs_ffts, axis=0)
    max_fft = np.zeros(blurred_ffts.shape[1:], dtype=complex)
    for i in range(dim):  # Sorry, didn't want to spend time optimizing this damn for-for-loops
        for j in range(dim):
            max_fft[i, j] = blurred_ffts[argmax_ffts[i, j], i, j]

    return max_fft


def estimate_fft_using_soft_max(blurred_ffts):
    abs_ffts = np.abs(blurred_ffts)
    min_abs_fft = np.min(abs_ffts)
    max_abs_fft = np.max(abs_ffts)
    range_fft = max_abs_fft - min_abs_fft + 1e-7
    abs_ffts_normalized = (abs_ffts - min_abs_fft) / range_fft
    weight = np.exp(10* abs_ffts_normalized)
    total_weight = np.sum(weight, axis=0)
    restored_fft = np.sum(weight / total_weight * blurred_ffts, axis=0)
    argmax = np.argmax(abs_ffts, axis=0)
    assert np.all(argmax == np.argmax(weight, axis=0))
    assert not np.any(np.isnan(weight))
    assert not np.any(np.isinf(weight))
    assert not np.any(np.isnan(total_weight))
    assert not np.any(np.isinf(total_weight))

    return restored_fft


def estimate_fft_using_weighted_abs_p_generator(p):
    def f(blurred_ffts):
        abs_ffts = np.abs(blurred_ffts) ** p
        weights = abs_ffts / np.sum(abs_ffts, axis=0)
        return np.sum(blurred_ffts * weights, axis=0)

    return f


def test_vertical_horizontal_perfect_deblur():
    im = read_original_image()
    v_k = 0.01
    motion_path_x = np.arange(0.0, 16.10, v_k)[np.newaxis, ...]
    motion_path_x -= np.mean(motion_path_x)
    motion_path_y = np.zeros(motion_path_x.shape)
    h_psf = generate_psfs(motion_path_x, motion_path_y, psf_fine)[0]
    v_psf = h_psf.T
    sanity_psf = np.zeros((19, 19), dtype=float)
    sanity_psf[9][9] = 1.0
    if (False):  # Uncomment if want to generate psfs directly
        v_psf = np.zeros((19, 19), dtype=float)
        n = 7  # we do 2*n - 1
        for i in range(n):
            v_psf[9 + i][9] = v_psf[9 - i][9] = 1 / (2 * n - 1)
        assert abs(np.sum(v_psf) - 1) < 1e-9

    blurred = forward_blur(im, [v_psf, v_psf.T])

    plt.subplot(3, 3, 1)
    plt.imshow(v_psf, cmap='gray')
    plt.title('vertical')

    plt.subplot(3, 3, 2)
    plt.imshow(v_psf.T, cmap='gray')
    plt.title('horizontal')

    plt.subplot(3, 3, 3)
    plt.imshow(sanity_psf, cmap='gray')
    plt.title('sanity')

    plt.subplot(3, 3, 4)
    plt.imshow(blurred[0], cmap='gray')
    plt.title('vertical')

    plt.subplot(3, 3, 5)
    plt.imshow(blurred[1], cmap='gray')
    plt.title('horizontal')

    plt.subplot(3, 3, 6)
    sanity_blurred = forward_blur(im, [sanity_psf])[0]
    plt.imshow(sanity_blurred, cmap='gray')
    plt.title('sanity')

    plt.subplot(3, 3, 7)
    plt.imshow(im, cmap='gray')
    plt.title('original')

    plt.subplot(3, 3, 8)
    deblur = inverse_deblur(blurred, estimate_fft_using_max)
    plt.imshow(deblur, cmap='gray')
    plt.title('de-blurred')

    plt.subplot(3, 3, 9)
    sanity_deblur = inverse_deblur([sanity_blurred], estimate_fft_using_max)
    plt.imshow(sanity_deblur, cmap='gray')
    plt.title('sanity de-blurred')

    plt.show()

    # assert sanity blur did nothing
    np.testing.assert_almost_equal(sanity_deblur, im)

    # assert sanity deblur did nothing
    np.testing.assert_almost_equal(sanity_blurred, im)

    # when the sanity blurred image (original) is in the samples, we really restore the original image.
    np.testing.assert_almost_equal(inverse_deblur(blurred + [sanity_blurred], estimate_fft_using_max), im)

    # assert h + v blurs result perfect de-blurring
    np.testing.assert_almost_equal(deblur, im)


if __name__ == '__main__':
    test_vertical_horizontal_perfect_deblur()
