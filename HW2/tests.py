import numpy as np
from scipy.signal import convolve2d
from unittest import TestCase

from point_spread_function import kernel_to_doubly_block_circulant, construct_blur_kernel_spatial


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
