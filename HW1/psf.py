import math

import numpy as np
import matplotlib.pyplot as plt


def psf_coarse(x, y, psf):
    x, y = int(round(x)), int(round(y))
    psf[y, x] += 1


def psf_fine(x, y, psf):
    # we assume there is a shift and it is not an integer. easier to compute this way.
    if math.floor(x) == math.ceil(x):
        x += 1e-7
        assert math.floor(x) + 1 == math.ceil(x)
    if math.floor(y) == math.ceil(y):
        y += 1e-7
        assert math.floor(y) + 1 == math.ceil(y)
    # say x,y = (0.4,0.3)
    top_left_idx = int(math.floor(x)), int(math.floor(y))  # (0,0), we floor for top since y starts from above
    top_right_idx = int(math.ceil(x)), int(math.floor(y))  # (1,0)
    bottom_left_idx = int(math.floor(x)), int(math.ceil(y))  # (0,1)
    bottom_right_idx = int(math.ceil(x)), int(math.ceil(y))  # (1,1)
    psf[top_left_idx[::-1]] += (bottom_right_idx[0] - x) * (bottom_right_idx[1] - y)  # 0.6*0.7
    psf[top_right_idx[::-1]] += (1 - (bottom_right_idx[0] - x)) * (bottom_right_idx[1] - y)  # 0.4*0.7
    psf[bottom_left_idx[::-1]] += (bottom_right_idx[0] - x) * (y - top_left_idx[1])  # 0.6*0.3
    psf[bottom_right_idx[::-1]] += (1 - (bottom_right_idx[0] - x)) * (y - top_left_idx[1])  # 0.4*0.3


def generate_psfs(X, Y, granularity_method):
    max_shift = math.ceil(max(np.max(np.abs(X)), np.max(np.abs(Y))))  # this is the maximal shift a camera has.
    rect_size = int(2 * (max_shift + 1)) + 1  # this ensures that the shift is within the rect and a center is defined.
    num_psfs, num_samples_per_motion = X.shape
    print("Generating {} PSFs of size: {}x{} each".format(num_psfs, rect_size, rect_size))
    center_offset = max_shift  # convenience.
    X = X + center_offset
    Y = Y + center_offset
    psfs = list()
    for i in range(num_psfs):
        psf = np.zeros((rect_size, rect_size), dtype=float)
        for x, y in zip(X[i], Y[i]):
            granularity_method(x, y, psf)
        psf /= num_samples_per_motion
        psfs.append(psf)
    return psfs


def test_psf_fine():
    x = 5.06
    y = 4.1
    psf = np.zeros((10, 10))
    psf_fine(x, y, psf)
    np.testing.assert_almost_equal(psf[4, 5], 0.94 * 0.9)  # top left
    np.testing.assert_almost_equal(psf[4, 6], 0.06 * 0.9)  # top right
    np.testing.assert_almost_equal(psf[5, 5], 0.94 * 0.1)  # bottom left
    np.testing.assert_almost_equal(psf[5, 6], 0.06 * 0.1)  # bottom right


def plot_test_psf_adv():
    X = np.arange(100).reshape((1, 100))
    X[0, -1] = 0.0
    Y = np.zeros(100).reshape((1, 100))
    psf = generate_psfs(X, Y, psf_fine)[0]
    plt.imshow(psf)
    plt.show()
    pass


if __name__ == '__main__':
    test_psf_fine()
    # plot_test_psf_adv()
