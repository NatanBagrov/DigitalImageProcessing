import cv2
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr


def visualize_multiple_common(dim, ax_index_action, file_name, title=None, show=True):
    fig = plt.figure(figsize=(dim, dim))
    axes = [fig.add_subplot(dim, dim, r * dim + c + 1) for r in range(0, dim) for c in range(0, dim)]
    for i, ax in enumerate(axes):
        ax_index_action(ax, i)
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None:
        fig.suptitle(title)
    if show:
        fig.show()
    fig.savefig(os.path.join('plots', file_name))
    fig.clf()


def visualize_motion_paths(X, Y, show=True):
    num_motions, num_samples_per_motion = X.shape
    dim = int(math.ceil(math.sqrt(num_motions)))
    colors = np.arange(num_samples_per_motion, dtype=int)
    visualize_multiple_common(dim, lambda ax, i: ax.scatter(X[i], Y[i], c=colors, s=0.05),
                              'motion_paths', 'Generated Motion Paths', show)


def visualize_psfs(psfs, show=True):
    num_psfs = len(psfs)
    dim = int(math.ceil(math.sqrt(num_psfs)))
    visualize_multiple_common(dim, lambda ax, i: ax.imshow(cv2.flip(psfs[i], 0), cmap='gray'),  # Note the vert flip.
                              'psfs', 'Generated PSFs', show)


def visualize_images(images, directory, file_name_prefix, title_prefix='', show=False):
    for i, image in enumerate(images):
        plt.imshow(image)
        plt.axis('off')
        plt.title('{} #{}'.format(title_prefix, i))
        plt.savefig(os.path.join(directory, '{}_{}'.format(file_name_prefix, i)))
        if show:
            plt.show()
        plt.clf()


def visualize_psnrs(original_image, blurred_images, deblurred_images, show=False):
    blurred_psnrs = np.array([compare_psnr(original_image, blurred_image) for blurred_image in blurred_images])
    deblurred_psnrs = np.array([compare_psnr(original_image, deblurred_image) for deblurred_image in deblurred_images])
    plt.plot(blurred_psnrs, label="blurred")
    plt.plot(deblurred_psnrs, label="deblurred")
    plt.legend()
    plt.title("accumulated deblurred PSNR vs. blurred PSNR")
    plt.xlabel("blurred: image-wise (image index),\ndeblurred: accumulated number of images")
    plt.ylabel("PSNR value")
    plt.savefig(os.path.join('plots', 'psnr_comparison'))
    if show:
        plt.show()
    plt.clf()
