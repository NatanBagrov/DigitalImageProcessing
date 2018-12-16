from forward_inverse_problem import forward_blur, inverse_deblur, estimate_fft_using_max, \
    estimate_fft_using_weighted_abs_p_generator
from psf import generate_psfs, psf_fine
from readers import read_motion_paths, read_original_image
from visualization import visualize_images, visualize_psfs, visualize_psnrs
import numpy as np

if __name__ == '__main__':
    X, Y = read_motion_paths()
    # visualize_motion_paths(X, Y, show=True)

    psfs = generate_psfs(X, Y, psf_fine)
    # visualize_psfs(psfs, show=True)

    orig_image = read_original_image()

    blurred_images = forward_blur(orig_image, psfs)
    blurred_images = \
        list(map(lambda x: np.around(x).astype(int), blurred_images))  # I guess we should consider integers
    # visualize_images(blurred_images, directory='blurred', file_name_prefix='blurred', title_prefix='Blurred')

    deblurred_images = [inverse_deblur(blurred_images[:i + 1], estimate_fft_using_weighted_abs_p_generator(p=3))
                        for i in range(len(blurred_images))]
    deblurred_images = \
        list(map(lambda x: np.around(x).astype(int), deblurred_images))  # I guess we should consider integers
    # visualize_images(deblurred_images, directory='deblurred', file_name_prefix='deblurred', title_prefix='De Blurred')

    visualize_psnrs(orig_image, blurred_images, deblurred_images, show=True)
