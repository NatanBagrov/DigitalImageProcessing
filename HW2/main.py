import logging

import numpy as np
import cv2
from scipy.fftpack import fftshift
from skimage.measure import compare_psnr

from deblur import \
    inverse_filter_1, \
    estimate_high_resolution_image, \
    bilinear_kernel, \
    bicubic_kernel, \
    total_variation_de_blurring
from point_spread_function import \
    gaussian_point_spread_function, \
    box_point_spread_function, \
    get_low_and_high_resolution_point_spread_function, \
    apply_point_spread_function_spatial, \
    construct_blur_kernel_spatial
from visualization import plot_images_grid


def range_invariant_compare_psnr(image_true, image_predicted):
    min_true = np.min(image_true)
    max_true = np.max(image_true)
    min_predicted = np.min(image_predicted)
    max_predicted = np.max(image_predicted)

    assert min_predicted < max_predicted

    image_predicted = \
        (image_predicted - min_predicted) \
        * (max_true - min_true) \
        / (max_predicted - min_predicted) \
        + min_true

    return compare_psnr(image_true, image_predicted)


def main():
    image_file_path = 'DIPSourceHW2.png'
    sigma = 4.0
    size = 4
    alpha = 2.5
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger = logging.getLogger()
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)

    # plt.ion()
    image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE) / 255.0
    print(image.shape, image.dtype, image.max(), image.min())
    plot_images_grid([[image]], title='Continuous')
    psf_high_height, psf_high_width = 32, 32

    name_to_function_and_args = {
        'gaussian kernel': (gaussian_point_spread_function, (sigma, psf_high_height, psf_high_width)),
        'box function': (box_point_spread_function, (size, psf_high_height, psf_high_width)),
    }

    # 1 and 2
    logger.debug('psfs')
    name_to_psfs = {
        name: get_low_and_high_resolution_point_spread_function(
            point_spread_function,
            alpha,
            *args
        )
        for name, (point_spread_function, args) in name_to_function_and_args.items()
    }

    plot_images_grid(
        list(name_to_psfs.values()),
        title='spatial psf',
        titles=[
            ['{} low'.format(name), '{} high'.format(name)]
            for name in name_to_psfs.keys()
        ],
        file_path_to_save='deliverable/psf.png',

    )
    plot_images_grid(
        list(name_to_psfs.values()),
        title='frequency psf',
        titles=[
            ['{} low'.format(name), '{} high'.format(name)]
            for name in name_to_psfs.keys()
        ],
        to_frequency=True,
    )

    # 3
    logger.debug('images')
    name_to_images = {
        name: [
            apply_point_spread_function_spatial(point_spread_function, image)
            for point_spread_function in point_spread_functions
        ]
        for name, point_spread_functions in name_to_psfs.items()
    }

    plot_images_grid(
        list(name_to_images.values()),
        title='images',
        titles=[
            ['{} low'.format(name), '{} high'.format(name)]
            for name in name_to_images.keys()
        ],
        file_path_to_save='deliverable/images.png',
    )

    # 4
    logger.debug('ks')
    name_to_k = {
        name: construct_blur_kernel_spatial(*point_spread_functions, dense=True)
        for name, point_spread_functions in name_to_psfs.items()
    }

    plot_images_grid(
        [list(name_to_k.values())],
        title='k',
        titles=[[
            name
            for name in name_to_k.keys()
        ]],
        file_path_to_save='deliverable/k.png',
    )

    logger.debug('estimated_psf_low')
    name_to_estimated_psf_low = {
        name: apply_point_spread_function_spatial(k, name_to_psfs[name][1])
        for name, k in name_to_k.items()
    }

    plot_images_grid([list(name_to_estimated_psf_low.values())], title='spatial estimated PSF_L')

    logger.debug('estimated_image_low')
    name_to_estimated_image_low = {
        name: apply_point_spread_function_spatial(k, name_to_images[name][1])
        for name, k in name_to_k.items()
    }

    plot_images_grid([list(name_to_estimated_psf_low.values())], title='spatial estimated low images from high')

    # 5
    # i
    logger.debug('inverse_filter_estimation')
    name_to_inverse_filter_estimation = {
        name: inverse_filter_1(name_to_images[name][0], k)
        for name, k in name_to_k.items()
    }
    name_to_inverse_filter_psnr = {
        name: range_invariant_compare_psnr(name_to_images[name][1], estimation)
        for name, estimation in name_to_inverse_filter_estimation.items()
    }

    plot_images_grid(
        [list(name_to_inverse_filter_estimation.values())],
        title='Wiener',
        titles=[[
            '{}\nPSNR:{:.2f}'.format(name, name_to_inverse_filter_psnr[name])
            for name in name_to_k.keys()
        ]],
        file_path_to_save='deliverable/wiener.png',
    )

    # ii
    logger.debug('tv_estimate')
    name_to_tv_estimate = {
        name: total_variation_de_blurring(
            name_to_images[name][0],
            k,
            5e4,
            10,
            1.618,
            1e-3,
        )
        for name, k in name_to_k.items()
    }
    name_to_tv_psnr = {
        name: range_invariant_compare_psnr(name_to_images[name][1], estimation)
        for name, estimation in name_to_tv_estimate.items()
    }

    plot_images_grid(
        [list(name_to_tv_estimate.values())],
        title='TV',
        titles=[[
            '{}\nPSNR:{:.2f}'.format(name, name_to_tv_psnr[name])
            for name in name_to_k.keys()
        ]],
        file_path_to_save='deliverable/tv.png',
    )

    # 6
    plot_images_grid([[bicubic_kernel(), bilinear_kernel()]], title='kernels')
    logger.debug('kernels')
    method_to_name_to_image = {
        method: {
            name: estimate_high_resolution_image(images[0], kernel())
            for name, images in name_to_images.items()
        }
        for kernel, method in (
            (bilinear_kernel, 'bilinear'),
            (bicubic_kernel, 'bicubic'),
        )
    }
    method_to_name_to_psnr = {
        method: {
            name: range_invariant_compare_psnr(name_to_images[name][1], estimation)
            for name, estimation in name_to_image.items()
        }
        for method, name_to_image in method_to_name_to_image.items()
    }

    plot_images_grid(
        list(map(list, map(dict.values, method_to_name_to_image.values()))),
        title='h',
        titles=[
            [
                '{} -> {}\nPSNR:{:.2f}'.format(psf_kernel, method, method_to_name_to_psnr[method][psf_kernel])
                for psf_kernel in name_to_images.keys()
            ]
            for method, name_to_images in method_to_name_to_image.items()
        ],
        file_path_to_save='deliverable/interpolation.png'
    )

    plot_images_grid(
        [
            [
                name_to_images[psf_kernel][0],
                name_to_inverse_filter_estimation[psf_kernel],
                name_to_tv_estimate[psf_kernel],
            ]
            +
            [
                name_to_estimate[psf_kernel]
                for method, name_to_estimate in method_to_name_to_image.items()
            ]
            +
            [
                name_to_images[psf_kernel][1],
            ]
            for psf_kernel in name_to_function_and_args.keys()
        ],
        titles=[
            [
                'PSNR:{:.2f}'.format(range_invariant_compare_psnr(name_to_images[psf_kernel][1], name_to_images[psf_kernel][0])),
                'PSNR:{:.2f}'.format(name_to_inverse_filter_psnr[psf_kernel]),
                'PSNR:{:.2f}'.format(name_to_tv_psnr[psf_kernel]),
            ]
            +
            [
                'PSNR:{:.2f}'.format(name_to_psnr[psf_kernel])
                for method, name_to_psnr in method_to_name_to_psnr.items()

            ]
            +
            [
                'PSNR:{:.2f}'.format(range_invariant_compare_psnr(name_to_images[psf_kernel][1], name_to_images[psf_kernel][1])),
            ]
            for psf_kernel in name_to_function_and_args.keys()
        ],
        row_headers=list(name_to_function_and_args.keys()),
        column_headers=(['low', 'Wiener', 'TV', ] + list(method_to_name_to_image.keys()) + ['high']),
        disable_ticks=True,
        file_path_to_save='deliverable/all.png'
    )

    debug = 42


if __name__ == '__main__':
    main()
