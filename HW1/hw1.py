import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import compare_psnr

from forward_inverse_problem import forward_blur, inverse_deblur
from psf import generate_psfs, psf_fine
from readers import read_motion_paths, read_original_image
from visualization import visualize_motion_paths, visualize_psfs, visualize_images

if __name__ == '__main__':
    X, Y = read_motion_paths()
    # visualize_motion_paths(X, Y, show=True)

    psfs = generate_psfs(X, Y, psf_fine)
    # visualize_psfs(psfs, show=True)

    orig_image = read_original_image()

    blurred_images = forward_blur(orig_image, psfs)
    blurred_images = list(map(lambda x: x.astype(int), blurred_images))  # I guess we should consider integers
    # visualize_images(blurred_images, directory='blurred', file_name_prefix='blurred', title_prefix='Blurred')

    deblurred_images = [inverse_deblur(blurred_images[:i + 1]) for i in range(len(blurred_images))]
    deblurred_images = list(map(lambda x: x.astype(int), deblurred_images))  # I guess we should consider integers
    visualize_images(deblurred_images, directory='deblurred', file_name_prefix='deblurred', title_prefix='De Blurred')

    blurred_psnrs = [compare_psnr(orig_image, blurred_image) for blurred_image in blurred_images]
    deblurred_psnrs = [compare_psnr(orig_image, deblurred_image) for deblurred_image in deblurred_images]
