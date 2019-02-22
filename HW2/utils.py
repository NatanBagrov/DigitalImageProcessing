import numpy as np
from scipy.signal import correlate2d, convolve2d


def conjugate_transpose(array):
    return np.transpose(np.conjugate(array))


def correlate(f, h):
    return correlate2d(f, h, mode='same', boundary='wrap')


def convolve(f, h):
    return convolve2d(f, h, mode='same', boundary='wrap')
