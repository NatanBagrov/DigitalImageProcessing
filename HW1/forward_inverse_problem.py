from itertools import combinations

import cv2
import numpy as np
from numpy.fft import fft2
from scipy import fftpack


def forward_blur(original_image, psfs):
    print("Generating {} blurred images (from PSFs)".format(len(psfs)))
    return [cv2.filter2D(original_image, -1, psf) for psf in psfs]


def inverse_deblur(blurred_images):
    print("Generating a de-blurred image using {} blurred images".format(len(blurred_images)))
    blurred_ffts = np.array([fft2(im) for im in blurred_images])
    abs_ffts = np.abs(blurred_ffts)
    argmax_ffts = np.argmax(abs_ffts, axis=0)
    max_fft = np.zeros(blurred_ffts.shape[1:], dtype=complex)
    for i in range(argmax_ffts.shape[0]):  # Sorry, didn't want to spend time optimizing this damn for-for-loops
        for j in range(argmax_ffts.shape[1]):
            max_fft[i, j] = blurred_ffts[argmax_ffts[i, j], i, j]

