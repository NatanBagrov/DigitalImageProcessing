import os

import cv2
import numpy as np


def read_motion_paths():
    import scipy.io as sio
    path = os.path.join('provided', '100_motion_paths.mat')
    print("Reading motion paths from: %s" % path)
    mat = sio.loadmat(path)
    return np.array(mat['X']), np.array(mat['Y'])


def read_original_image():
    path = os.path.join('provided', 'DIPSourceHW1.jpg')
    print("Reading original image from: %s" % path)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
