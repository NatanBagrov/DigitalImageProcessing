from unittest import TestCase

import numpy as np
from scipy.signal import convolve2d
from scipy.optimize import minimize
from skimage import data
from skimage.transform import resize
from skimage.measure import compare_mse
import matplotlib.pyplot as plt
import cv2

from point_spread_function import \
    kernel_to_doubly_block_circulant, \
    construct_blur_kernel_spatial, \
    gaussian_point_spread_function, \
    gradient_doubly_block_circulant
from optimization import alternating_direction_method_of_multipliers, de_degradation_f_step_multiplication
from deblur import total_variation_de_blurring_spatial, total_variation_de_blurring_frequency, total_variation_de_blurring, d
from visualization import plot_images_grid
from deblur import d, dt


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
            actual.todense(),
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
            actual.todense(),
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

        np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-5)

    def test_d(self):
        h = 5
        w = 4
        u = np.random.rand(h, w)

        dux_expected = np.hstack((np.diff(u, 1, 1), np.reshape(u[:, 0] - u[:, -1], (-1, 1))))
        duy_expected = np.vstack((np.diff(u, 1, 0), np.reshape(u[0, :] - u[-1, :], (1, -1))))

        dux_actual, duy_actual = d(u)

        np.testing.assert_allclose(dux_actual, dux_expected)
        np.testing.assert_allclose(duy_actual, duy_expected)

        dtxy_expected = np.hstack((
            np.reshape(dux_expected[:, -1] - dux_expected[:, 0], (-1, 1)),
            -np.diff(dux_expected, 1, 1),
        )) + np.vstack((
            np.reshape(duy_expected[-1, :] - duy_expected[0, :], (1, -1)),
            -np.diff(dux_expected, 1, 0),
        ))
        dtxy_actual = dt(dux_actual, duy_actual)

        np.testing.assert_allclose(dtxy_actual, dtxy_expected)

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
        # image = self.__class__.load('camera')
        image = cv2.imread('/Applications/MATLAB_R2018b.app/toolbox/images/imdata/cameraman.tif', 0) / 255.0
        # kernel = gaussian_point_spread_function(10, 4, 4)
        kernel = np.ones((15, 15)) / 15.0 / 15.0

        # scipy_result = self.__class__.do_scipy(image, kernel)
        # scipy_error = compare_mse(image, scipy_result)

        my_result = self.__class__.do_my(image, kernel)
        my_error = compare_mse(image, my_result)

        # self.assertGreaterEqual(scipy_error, my_error)

    @staticmethod
    def load(name):
        return resize(getattr(data, name).__call__() / 255.0, (256, 256), mode='wrap')

    @staticmethod
    def do_my(image, kernel, lambda_=1e-2):
        def callback(f):
            dx, dy = d(f)
            gradient = np.sqrt(np.square(dx) + np.square(dy))
            print('L2: {}\t TV: {}\t ADM MSE={}'.format(
                np.linalg.norm(blurred_image - convolve2d(f, kernel, mode='same', boundary='wrap'), ord=2),
                np.linalg.norm(gradient, ord=1),
                compare_mse(image, f)
            ))
            plt.imshow(f.reshape(image.shape), cmap='gray')
            plt.title('ADM')
            plt.show()

            return False

        blurred_image = convolve2d(image, kernel, mode='same', boundary='wrap')
        rho = 1.0 / 5000.0
        epsilon = 1e-9
        restored_image = total_variation_de_blurring(
            blurred_image,
            kernel,
            5e4,
            10,
            1.618,
            1e-3,
            callback=callback,
        )
        plot_images_grid([[image, blurred_image], [kernel, restored_image]])

        return restored_image

    @staticmethod
    def do_scipy(image, kernel, lambda_=1e-2):
        blurred_image = convolve2d(image, kernel, mode='full', boundary='fill', fillvalue=0)

        def callback(f):
            f = f.reshape(image.shape)
            print('SCIPY MSE={}'.format(compare_mse(image, f)))
            plt.imshow(f.reshape(image.shape), cmap='gray')
            plt.title('SCIPY')
            plt.show()

            return False

        def objective(f):
            f = f.reshape(image.shape)
            blurred_image_estimate = convolve2d(f, kernel, mode='full', boundary='fill', fillvalue=0)
            loss_l2 = np.linalg.norm(blurred_image - blurred_image_estimate, ord=2)
            gradient_x, gradient_y = np.gradient(f)
            gradient_abs = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
            loss_tv = np.linalg.norm(gradient_abs, ord=1)

            return loss_l2 + lambda_ * loss_tv

        return minimize(
            objective,
            np.random.rand(*image.shape).flatten(),
            tol=1e-3,
            callback=callback,
        ).x.reshape(image.shape)
