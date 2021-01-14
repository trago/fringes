import numpy as np
from typing import List, Tuple, Union, Any

from skimage.filters import gaussian
from . import create_matrix
from .psi_vu import calc_term_V, calc_term_U, print_report_info


def vu_factorization(matrix_I: np.ndarray, error_accuracy: float = 1e-3,
                     max_iters: int = 20, matrix_V: np.ndarray = None,
                     verbose: bool = False, verbose_step: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the factorization :math:`\mathbf V\mathbf U` of phase-shifting fringe patterns given the matrix
    :math:`\mathbf I`.

    :param matrix_I: the matrix :math:`\mathbf I`
    :param error_accuracy: the accuracy convergence
    :param max_iters: maximum number of iterations given if accuracy is not reached
    :param verbose: if want to print text messages in output
    :param verbose_step: messages are printed each step iterations
    :return: a tupla with two matrices. The first matrix is :math:`\mathbf V` and the second is :math:`\mathbf U`
    """
    matrix_I += 1
    M, N = matrix_I.shape
    deltas = np.linspace(0, np.pi, N - 2, endpoint=False)

    if matrix_V is not None:
        matrix_U = calc_term_U(matrix_I, matrix_V)
        matrix_V = calc_term_V(matrix_I, matrix_U)
    else:
        matrix_U = np.array([(1, 0, 1, 0), (0, 1, 0, 1), (0, 0, np.cos(deltas[0]), np.cos(deltas[1])),
                             (0, 0, np.sin(deltas[0]), np.sin(deltas[1]))]).T
        matrix_V = calc_term_V(matrix_I, matrix_U)
    previous_phase = deltas[1]

    error = 1.0
    iter = 1
    for iter in range(1, max_iters):
        phase, _ = calc_phase(matrix_V[:, 1:])
        # matrix_V[:, 0] = np.ones_like(phase)
        matrix_V[:, 2] = np.cos(phase)
        matrix_V[:, 3] = -np.sin(phase)
        matrix_U = calc_term_U(matrix_I, matrix_V)

        steps, _ = calc_phase(matrix_U[:, 1:])
        # matrix_U[:, 0] = np.ones_like(deltas)
        matrix_U[:, 2] = np.array((0, 0, np.cos(steps[2]), np.cos(steps[3])))
        matrix_U[:, 3] = np.array((0, 0, np.sin(steps[2]), np.sin(steps[3])))
        matrix_V = calc_term_V(matrix_I, matrix_U)

        step = np.abs(steps[2] - steps[3])
        error = np.abs(step - previous_phase)
        previous_phase = step

        if error < error_accuracy:
            break
        # if verbose:
        # print_iter_info(iter, error, error_accuracy, verbose_step)

    if verbose:
        print_report_info(iter, error, error_accuracy)

    return matrix_V, matrix_U


def calc_phase(matrix_V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the phase from the matrix matrix_V.

    :param matrix_V: the matrix :math:`\mathbf V`
    :return: a 1D array with the phase
    """
    return np.arctan2(-matrix_V[:, 1], matrix_V[:, 2]), matrix_V[:, 0]


def calc_phase_2steps(matrix_I, delta):
    """
    2-steps phase-shifting.
    It receives a list of two interferograms and the phase-shifting value to compute its modulating phase.

    :param matrix_I: A Nx2 matrix of fringe patterns. Each column is a fringe pattern
    :param delta: the phase-shifting value
    :return: A
    """
    matrix_Up = np.array([(1, 1, 1), (0, np.cos(0), np.cos(delta)), (0, np.sin(0), np.sin(delta))])
    inv_Up = np.linalg.inv(matrix_Up)
    matrix_Vp = matrix_I @ inv_Up

    cc = matrix_Vp[:, 1]
    ss = matrix_Vp[:, 2]

    pp = np.arctan2(-ss, cc)
    return pp


def linear_map(image, min, max, a, b):
    return (image - min) * (b - a) / (max - min) + a


def vec_to_mask(mask: np.ndarray, vec_img):
    image = np.zeros(mask.shape)
    image[mask == 1] = vec_img
    return image


def create_matrix(dc, image_list, mask: np.ndarray = None):
    if (len(image_list) == 2):
        if isinstance(image_list, tuple):
            image_list = dc + image_list
        elif isinstance(image_list, list):
            image_list = [dc[0], dc[1]] + image_list
        else:
            raise TypeError("Parameter image_list: A list or tuple is expected")
    elif len(image_list) > 3:
        image_list[0] = dc[0]
        image_list[1] = dc[1]
    else:
        raise IndexError(f"Parameter image_list: len(image_list) must be >= 2, current value = {len(image_list)}")
    if mask is None:
        image_list = map(lambda image: image.flatten(), image_list)
    else:
        image_list = map(lambda image: apply_mask_to_vec(image, mask), image_list)
    matrix = np.vstack(tuple(image_list)).T

    return matrix


def apply_mask_to_vec(image: np.ndarray, mask: np.ndarray):
    image = image[mask == 1]
    return image


def img_reshape(mask: Union[Tuple[int, int], np.ndarray], img: np.ndarray) -> np.ndarray:
    if isinstance(mask, tuple) or isinstance(mask, list):
        return img.reshape(mask)
    return vec_to_mask(mask, img)


def approximate_dc(image_list, size: float = 64.0) -> Tuple[Any, ...]:
    ones_response = gaussian(np.ones_like(image_list[0]), size, mode='constant')

    dc = map(lambda image: gaussian(image, size, mode='constant') / ones_response, image_list)

    return tuple(dc)


def two_frames_demodulation(image_list: Union[List[np.ndarray], Tuple[np.ndarray]], delta, mask: np.ndarray = None,
                            dc_kernel: float = 64, blur_kernel: float = 0.5) -> np.ndarray:
    if len(image_list) < 2:
        raise IndexError("The image list mist have at least two images")

    images = map(lambda image: image.astype(dtype=float) if image.dtype != float else image,
                 image_list)
    images = tuple(images)
    for image in image_list:
        min = image.min()
        max = image.max()
        image[:, :] = linear_map(image, min, max, -1, 1)

    if blur_kernel > 0:
        images = map(lambda image: gaussian(image, blur_kernel), images)
    images = tuple(images)

    dc = approximate_dc(images, dc_kernel)
    image_matrix = create_matrix(dc, images, mask)

    return calc_phase_2steps(image_matrix, delta)


def two_frames_vu(image_list: Union[List[np.ndarray], Tuple[np.ndarray]], mask: np.ndarray = None,
                  dc_kernel: float = 64, blur_kernel: float = 0.5, vu_iters=50) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(image_list) < 2:
        raise IndexError("The image list mist have at least two images")

    images = map(lambda image: image.astype(dtype=float) if image.dtype != float else image,
                 image_list)
    images = tuple(images)
    for image in image_list:
        min = image.min()
        max = image.max()
        image[:, :] = linear_map(image, min, max, -1, 1)

    if blur_kernel > 0:
        images = map(lambda image: gaussian(image, blur_kernel), images)
    images = tuple(images)

    dc = approximate_dc(images, dc_kernel)
    image_matrix = create_matrix(dc, images, mask)

    mat_v, mat_u = vu_factorization(image_matrix, error_accuracy=1e-6, verbose=True, max_iters=vu_iters)
    pp = np.arctan2(-mat_v[:, 3], mat_v[:, 2])
    steps = np.arctan2(-mat_u[:, 3], mat_u[:, 2])

    if mask is None:
        shape = image_list[0].shape
        return pp.reshape(shape), steps[2:], mat_v[:, 0].reshape(shape), mat_v[:, 1].reshape(shape)
    else:
        pp = img_reshape(mask, pp)
        dc0 = img_reshape(mask, mat_v[:, 0])
        dc1 = img_reshape(mask, mat_v[:, 1])
        return pp, steps[2:], dc0, dc1
