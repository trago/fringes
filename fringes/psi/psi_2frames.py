import numpy as np
from typing import List, Tuple, Union
from .psi_vu import create_matrix, vu_factorization


def demodulate(image_list: List[np.ndarray], patch_size: int = 24) -> np.ndarray:
    """
    Given a list of fringe pattern images and the phase shift or step, recovers the modulating phase using the VU
    model.

    :param image_list: the list of fringe pattern images
    :param phase_step: the phase shift/step of the images.

                       It can be a list of phase step values or an scalar. If an scalar is given, the phase step of
                       each fringe pattern is given as :math:`\alpha n`, where :math:`\alpha` is the scalar and
                       :math:`n` is the :math:`n`-fringe pattern image.
    :param initial_step: the initial phase step
    :return: a 2D array with the demodulated phase
    """
    size_m = image_list[0].shape[0]
    size_n = image_list[0].shape[1]

    step = 0
    matrix_V: Union[np.ndarray, None] = None
    for m in range(0, size_m - patch_size, 4):
        for n in range(0, size_n - patch_size, 4):
            images = [image[m:m + patch_size, n:n * patch_size] for image in image_list]
            step_, dc, pp = demodulate_2steps(images, matrix_V)

    pp = calc_phase_2steps(img0, img1, step)

    return step, dc_, pp


def extend_images(image_list: List[np.ndarray]):
    image_list = [image_list[0][2:, 2:], image_list[1][2:, 2:],
                  image_list[0][0:-2, 0:-2], image_list[1][0:-2, 0:-2],
                  image_list[0][2:, :-2], image_list[1][2:, :-2],
                  image_list[0][:-2, 2:], image_list[1][:-2, 2:],
                  image_list[0][:-2, 2:], image_list[1][:-2, 2:],
                  image_list[0][1:-1, 1:-1], image_list[1][1:-1, 1:-1]]

    return image_list


def calc_phase_2steps(image0, image1, step):
    return np.arctan2(image0 * np.sin(step), image0 * np.cos(step) - image1)


def calc_step_dc(image_list: List[np.ndarray], matrix_V: np.ndarray = None, error_accuracy: float = 1e-3,
                 max_iters: int = 20,
                 verbose: bool = False, verbose_step: int = 5) -> np.ndarray:
    """
    Given a list of fringe pattern images, it recovers the phase using PSI-VU.

    It uses a factorization method called PSI-VU which works with phase-shifting fringe patterns.

    :param image_list: the list of fringe pattern images
    :param error_accuracy: the accuracy convergence of the factorization
    :param max_iters: maximum number of iterations
    :param verbose: if you want to print text messages with iteration info in the output
    :param verbose_step: print messages each *step* iteration
    :return: a 2D array with the obtained modulated phase
    """
    matrix_form = create_matrix(image_list)
    term_V, term_U = vu_factorization(matrix_form, error_accuracy, max_iters, matrix_V,
                                      verbose, verbose_step)
    steps = calc_shifts(term_U)
    a = term_U[:, 0].mean()

    return steps[1] - steps[0], term_V[:, 0].reshape(image_list[0].shape) * a - 1, term_V


def calc_shifts(matrix_U: np.ndarray) -> np.ndarray:
    """
    Computes the phase shifts from the matrix matrix_U.

    :param matrix_U: the matrix :math:`\mathbf U`
    :return: a 1D array with the phase shifts.
    """
    return np.arctan2(matrix_U[:, 2], matrix_U[:, 1])


def demodulate_2steps(image_list: List[np.ndarray], matrix_V: np.ndarray = None) -> np.ndarray:
    """
    Given a list of fringe pattern images and the phase shift or step, recovers the modulating phase using the VU
    model.

    :param image_list: the list of fringe pattern images
    :param phase_step: the phase shift/step of the images.

                       It can be a list of phase step values or an scalar. If an scalar is given, the phase step of
                       each fringe pattern is given as :math:`\alpha n`, where :math:`\alpha` is the scalar and
                       :math:`n` is the :math:`n`-fringe pattern image.
    :param initial_step: the initial phase step
    :return: a 2D array with the demodulated phase
    """
    images = extend_images(image_list)
    step, dc_, term_V = calc_step_dc(images, matrix_V, error_accuracy=1e-4, max_iters=50, verbose=True)

    return step, dc_, term_V
