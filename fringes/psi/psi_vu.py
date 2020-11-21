"""
Phase-shifting VU factorization.

.. moduleauthor:: Julio C. Estrada <julio@cimat.mx>
"""

import numpy as np
from typing import List, Tuple
import logging


def vu_factorization(matrix_I: np.ndarray, error_accuracy: float = 1e-3,
                     max_iters: int = 20,
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
    matrix_I += 10000
    M, N = matrix_I.shape
    step = 2 * np.pi / N
    initial_deltas = np.linspace(0, 2 * np.pi - step, N)
    s1_ones = np.ones(N)

    matrix_U = np.vstack([s1_ones, np.cos(initial_deltas), np.sin(initial_deltas)]).T
    matrix_V = calc_factor_V(matrix_I, matrix_U)
    previous_phase = calc_phase(matrix_V)

    error = 1.0
    iter = 1
    for iter in range(1, max_iters):
        matrix_V[:, 1] = np.cos(previous_phase)
        matrix_V[:, 2] = -np.sin(previous_phase)
        matrix_U = calc_factor_U(matrix_I, matrix_V)
        matrix_V = calc_factor_V(matrix_I, matrix_U)

        phase = calc_phase(matrix_V)
        error = np.sum(((previous_phase - phase) ** 2)) / float(M)
        previous_phase = phase

        if error < error_accuracy:
            break
        if verbose:
            print_iter_info(iter, error, error_accuracy, verbose_step)

    if verbose:
        print_report_info(iter, error, error_accuracy)

    return matrix_V, matrix_U


def calc_factor_U(matrix_I: np.ndarray, factor_V: np.ndarray) -> np.ndarray:
    """
    Given the matrices :math:`\mathbf I` and :math:`\mathbf V`, it computes the matrix :math:`\mathbf U`.
    :param matrix_I: the matrix :math:`\mathbf I`
    :param factor_V:
    :return:
    """
    aux_Ainv = np.linalg.inv(factor_V.T @ factor_V)
    factor_U = aux_Ainv @ factor_V.T @ matrix_I

    return factor_U.T


def calc_factor_V(matrix_I: np.ndarray, factor_U: np.ndarray) -> np.ndarray:
    """
    Given the matrices :math:`\mathbf I` and :math:`\mathbf U`, it computes the matrix :math:`\mathbf V`.

    :param matrix_I: the matrix :math:`\mathbf I`
    :param factor_U: the matrix :math:`\mathbf U`
    :return: a 2D array with the matrix :math:`\mathbf V`
    """
    aux_Binv = np.linalg.inv(factor_U.T @ factor_U)
    factor_V = matrix_I @ factor_U @ aux_Binv

    return factor_V


def create_matrix(image_list: List[np.ndarray]) -> np.ndarray:
    """
    Given a list of image :math:`N` images, it construct a matrix with dimension :math:`M\times N`.

    In this matrix, each column is an image from the list.

    :param image_list: the list of images
    :return: a matrix array :math:`M\times N`
    """
    shape = image_list[0].shape
    N = len(image_list)
    M = shape[0] * shape[1]
    matrix_images = np.zeros((M, N))

    for k, img in enumerate(image_list):
        if shape != img.shape:
            raise TypeError('Images in the list must have the same dimension')
        matrix_images[:, k] = img.reshape(1, shape[0] * shape[1])

    return matrix_images


def calc_phase(matrix_V: np.ndarray) -> np.ndarray:
    """
    Computes the phase from the matrix matrix_V.

    :param matrix_V: the matrix :math:`\mathbf V`
    :return: a 1D array with the phase
    """
    return np.arctan2(-matrix_V[:, 2], matrix_V[:, 1])


def calc_shifts(matrix_U: np.ndarray) -> np.ndarray:
    """
    Computes the phase shifts from the matrix matrix_U.

    :param matrix_U: the matrix :math:`\mathbf U`
    :return: a 1D array with the phase shifts.
    """
    return np.arctan2(matrix_U[:, 2], matrix_U[:, 1])


def calc_magnitude(matrix_psi: np.ndarray) -> np.ndarray:
    """
    Computes the magnitude of the last two columns of matrix matrix_psi.

    The magnitude is computed as

    .. math::

       p = \|V_{:,1} + iV_{:,2}\|

    where :math:`V` is the matrix, :math:`V_{:,1}` and :math:`V_{:,2}` are its second and third columns, respectively.

    :param matrix_psi: a :math:`M\times\3` matrix
    :return: a 2D array with the magnitude
    """

    return np.absolute(matrix_psi[:, 1] + 1j * matrix_psi[:, 2])


def print_iter_info(iter: int, error: float, error_tol: float,
                    verbose_step: int):
    """
    Prints iteration information in output
    :param iter: the iteration number
    :param error: the current relative error
    :param error_tol: the objective accuracy convergence error
    :param verbose_step: info text is printer each step iteration. This tell what is the step
    :return: None
    """
    if iter % verbose_step == 0:
        notif = '\t{0:04}: Objective accuracy: {1:10.3e} Current accuracy:{2:10.3e}'. \
            format(iter, error_tol, error)
        logging.info(notif)


def print_report_info(iter: int, error: float, error_tol: float):
    """
    Prints a summary of the iteration process.
    :param iter: the total iterations given
    :param error: the current relative error
    :param error_tol: the objective accuracy convergence error
    :return: None
    """
    if error < error_tol:
        print('Process finished with relative error = {0:10.4e}, iterations = {1}'.
              format(error, iter))
    else:
        print('Process finished by reaching the maximum number of iterations with error = {0:10.4e}'.
              format(error))


def demodulate(image_list: List[np.ndarray], error_accuracy: float = 1e-3,
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
    factor_V, factor_U = vu_factorization(matrix_form, error_accuracy, max_iters, verbose, verbose_step)

    return calc_phase(factor_V).reshape(image_list[0].shape)


def demodulate_psi(image_list: List[np.ndarray], phase_step: np.ndarray,
                   initial_step: int = 0.0) -> np.ndarray:
    """
    Given a list of fringe pattern images and the phase shift or step, recovers the modulating phase using the VU
    model.

    :param image_list: the list of fringe pattern images
    :param phase_step: the phase shift/step that images has.

                       It can be a list of phase step values or an scalar. If an scalar is given, the phase step of
                       each fringe pattern is given as :math:`\alpha n`, where :math:`\alpha` is the scalar and
                       :math:`n` is the :math:`n`-fringe pattern image.
    :param initial_step: the initial phase step
    :return: a 2D array with the demodulated phase
    """
    N = len(image_list)
    if isinstance(phase_step, list):
        deltas = np.array(phase_step) - initial_step
    elif isinstance(phase_step, np.ndarray):
        deltas = phase_step
    else:
        deltas = np.array([(n * phase_step - initial_step) for n in range(N)])

    image_matrix = create_matrix(image_list)

    s1_ones = np.ones(N)
    factor_U = np.vstack([s1_ones, np.cos(deltas), np.sin(deltas)]).T
    factor_V = calc_factor_V(image_matrix, factor_U)

    return calc_phase(factor_V).reshape(image_list[0].shape)
