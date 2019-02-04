from unittest import TestCase

import numpy as np
from scipy.signal import convolve2d
from skimage import data
from skimage.transform import resize
import matplotlib.pyplot as plt

from point_spread_function import \
    kernel_to_doubly_block_circulant, \
    construct_blur_kernel_spatial, \
    gaussian_point_spread_function
from optimization import alternating_direction_method_of_multipliers, de_degradation_f_step
from deblur import total_variation_de_blurring
from visualization import plot_images_grid


class PointSpreadFunctionTest(TestCase):
    def test_kernel_to_doubly_block_circulant_1(self):
        h = [[1, -1],
             [1, 1], ]
        actual = kernel_to_doubly_block_circulant(h, (2, 3))
        desired = [[1, 0, 0, 0, 0, 0, ],
                   [-1, 1, 0, 0, 0, 0, ],
                   [0, -1, 1, 0, 0, 0, ],
                   [0, 0, -1, 0, 0, 0, ],
                   [1, 0, 0, 1, 0, 0, ],
                   [1, 1, 0, -1, 1, 0, ],
                   [0, 1, 1, 0, -1, 1, ],
                   [0, 0, 1, 0, 0, -1, ],
                   [0, 0, 0, 1, 0, 0, ],
                   [0, 0, 0, 1, 1, 0, ],
                   [0, 0, 0, 0, 1, 1, ],
                   [0, 0, 0, 0, 0, 1, ]
                   ]
        np.testing.assert_allclose(
            actual,
            desired,
        )

        f = [[2, 5, 3],
             [1, 4, 1],]

        np.testing.assert_allclose(
            self.__class__.my_convolve(f, h),
            self.__class__.scipy_convolve(f, h),
        )

    def test_kernel_to_doubly_block_circulant_2(self):
        h = [[1, -1]]
        actual = kernel_to_doubly_block_circulant(h, (2, 2))
        desired = [[1, 0, 0, 0, ],
                   [-1, 1, 0, 0, ],
                   [0, -1, 0, 0, ],
                   [0, 0, 1, 0, ],
                   [0, 0, -1, 1, ],
                   [0, 0, 0, -1]
                   ]

        np.testing.assert_allclose(
            actual,
            desired
        )

        f = [[1, 2],
             [3, 4], ]

        np.testing.assert_allclose(
            self.__class__.my_convolve(f, h),
            self.__class__.scipy_convolve(f, h),
        )

    def test_construct_blur_kernel_spatial(self):
        height, widht = np.random.randint(1, high=64, size=2)
        psf_h = np.random.rand(height, widht)
        desired = np.random.rand(height, widht)
        psf_l = self.__class__.scipy_convolve(desired, psf_h)
        actual = construct_blur_kernel_spatial(psf_l, psf_h)

        np.testing.assert_allclose(actual, desired)

    @staticmethod
    def my_convolve(input1, input2):
        input1 = np.array(input1)
        input2 = np.array(input2)
        input2_dbs = kernel_to_doubly_block_circulant(input2, input1.shape)
        result_shape = np.array(input1.shape) + np.array(input2.shape) - 1
        output = np.reshape(input2_dbs @ np.ndarray.flatten(input1), result_shape)

        return output

    @staticmethod
    def scipy_convolve(input1, input2):
        input1 = np.array(input1)
        input2 = np.array(input2)

        return convolve2d(input1, input2, mode='full', boundary='fill', fillvalue=0)


class AlternatingDirectionMethodOfMultipliers(TestCase):
    def test_cameraman_gaussian_h_15_sigma_10(self):
        self.__class__.do(
            self.__class__.load('camera'),
            gaussian_point_spread_function(10, 15, 15)
        )

    @staticmethod
    def load(name):
        return resize(getattr(data, name).__call__() / 255.0, (128, 128))

    @staticmethod
    def do(image, kernel):
        def callback(f):
            plt.imshow(f.reshape(image.shape), cmap='gray')
            plt.show()

            return False

        blurred_image = convolve2d(image, kernel, mode='full', boundary='fill', fillvalue=0)
        rho = 1.0
        restored_image = total_variation_de_blurring(blurred_image, kernel,
                                                     calback=callback)
        plot_images_grid([[image, blurred_image], [kernel, restored_image]])

        return restored_image
