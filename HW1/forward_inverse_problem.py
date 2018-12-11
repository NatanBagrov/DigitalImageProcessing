import cv2
import numpy as np


def forward_blur(original_image, psfs):
    print("Generating {} blurred images (from PSFs)".format(len(psfs)))
    return [cv2.filter2D(original_image, -1, psf) for psf in psfs]


def inverse_deblur(blurred_images):
    print("Generating a de-blurred image using {} blurred images)".format(len(blurred_images)))
    raise NotImplementedError
