import cv2
import numpy as np
from scipy import fftpack


def forward_blur(original_image, psfs):
    print("Generating {} blurred images (from PSFs)".format(len(psfs)))
    return [cv2.filter2D(original_image, -1, psf) for psf in psfs]


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
