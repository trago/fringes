"""
Numerical simulation for testing the PSI-VU factorization.
"""

import numpy as np
from matplotlib import pylab as plt
from fringes.psi import *
from fringes.simulations import functions
from skimage.filters import gaussian
import skimage.io as ski_io
from fringes.psi import psi_vu2
from typing import Union, List, Tuple


def linear_map(image, min, max, a, b):
    return (image - min) * (b - a) / (max - min) + a


def apply_mask_to_vec(image: np.ndarray, mask: np.ndarray):
    image = image[mask == 1]
    return image


def vec_to_mask(mask: np.ndarray, vec_img):
    image = np.zeros(mask.shape)
    image[mask == 1] = vec_img
    return image


def create_matrix(dc, image_list, mask: np.ndarray=None):
    if (len(image_list) == 2):
        if isinstance(image_list, tuple):
            image_list = (dc,) + image_list
        elif isinstance(image_list, list):
            image_list = [dc] + image_list
        else:
            raise TypeError("Parameter image_list: A list or tuple is expected")
    elif len(image_list) > 2:
        image_list[0] = dc
    else:
        raise IndexError(f"Parameter image_list: len(image_list) must be >= 2, current value = {len(image_list)}")
    if mask is None:
        image_list = map(lambda image: image.flatten(), image_list)
    else:
        image_list = map(lambda image: apply_mask_to_vec(image, mask), image_list)
    matrix = np.vstack(tuple(image_list)).T

    return matrix


def img_reshape(mask: Union[Tuple[int, int], np.ndarray], img: np.ndarray) -> np.ndarray:
    if isinstance(mask, tuple) or isinstance(mask, list):
        return img.reshape(mask)
    return vec_to_mask(mask, img)


def calc_phase(matrix_I, delta):
    matrix_Up = np.array([(1, 1, 1), (0, np.cos(0), np.cos(delta)), (0, np.sin(0), np.sin(delta))])
    inv_Up = np.linalg.inv(matrix_Up)
    matrix_Vp = matrix_I @ inv_Up

    cc = matrix_Vp[:, 1]
    ss = matrix_Vp[:, 2]

    pp = np.arctan2(-ss, cc)
    magn = np.sqrt(ss ** 2 + cc ** 2)
    return magn * np.cos(pp), magn * np.sin(pp), pp


def approximate_dc(image_list, size=64) -> np.ndarray:
    ones_response = gaussian(np.ones_like(image_list[0]), size, mode='constant')

    dc = gaussian(image_list[0], size, mode='constant') / ones_response
    for n in range(1, len(image_list)):
        dc += gaussian(image_list[n], size, mode='constant') / ones_response

    return dc / len(image_list)


def two_frames_phase(image_list: Union[List[np.ndarray], Tuple[np.ndarray]], mask: np.ndarray=None,
                     dc_kernel: float = 64, blur_kernel: float = 0.5, vu_iters=50) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(image_list) < 2:
        raise IndexError("The image list mist have at least two images")

    images = map(lambda image: image.astype(dtype=float) if image.dtype != float else image,
                 image_list)
    if blur_kernel > 0:
        images = map(lambda image: gaussian(image, blur_kernel), images)
    images = tuple(images)

    dc = approximate_dc(images, dc_kernel)
    image_matrix = create_matrix(dc, images, mask)

    mat_v, mat_u = psi_vu2.vu_factorization(image_matrix, error_accuracy=1e-6, verbose=True, max_iters=vu_iters)
    pp = np.arctan2(-mat_v[:, 2], mat_v[:, 1])
    steps = np.arctan2(-mat_u[:, 2], mat_u[:, 1])

    if mask is None:
        shape = image_list[0].shape
        return pp.reshape(shape), steps[1:], mat_v[:, 0].reshape(shape)
    else:
        pp = img_reshape(mask, pp)
        dc = img_reshape(mask, mat_v[:, 0])
        return pp, steps[1:], dc


# Number of fringe patterns
K = 3
shape = (256, 512)
# Generating the phase shifts
# delta = np.random.rand(K) * 2 * np.pi
delta = [0.0, 1.57]
print(delta)

phase = functions.ramp(shape[0], shape[1], 6., 1)
phase = functions.peaks(shape[0], shape[1]) * 20
phase = functions.parabola(shape[0], shape[1]) * 0.00008
dc = 5 - functions.gaussian(200, phase.shape) * 10
contrast = 5.0 + 15 * functions.gaussian(140, phase.shape) + 1
noise = .5

image_list = [dc + contrast * np.cos(phase + d) + np.random.randn(*shape) * noise for d in delta]
I1 = ski_io.imread('../data/I1.png', as_gray=True)
I2 = ski_io.imread('../data/I2.png', as_gray=True)
mask = ski_io.imread('../data/p1_mask.png', as_gray=True)
image_list = (I1, I2)

pp, steps, dc_ = two_frames_phase(image_list, mask=None, dc_kernel=23, blur_kernel=3, vu_iters=100)

cc = np.cos(pp)
ss = np.sin(pp)

img0 = cc
img1 = ss

print(f"img0: ({img0.min(), img0.max()})")
print(f"img1: ({img1.min(), img1.max()})")
print(f"dc_: ({dc_.min(), dc_.max()})")
print(f"steps: {steps}")

plt.figure()
plt.subplot(121)
plt.imshow(image_list[0], cmap=plt.cm.gray)
plt.subplot(122)
plt.imshow(image_list[1], cmap=plt.cm.gray)

plt.figure()
plt.subplot(121)
plt.imshow(pp, cmap=plt.cm.gray)
plt.subplot(122)
plt.imshow(dc_, cmap=plt.cm.gray)

plt.show()
