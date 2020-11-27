import numpy as np
from typing import List, Tuple, Union
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
    deltas = np.linspace(0, np.pi, N-1, endpoint=False)

    if matrix_V is not None:
        matrix_U = calc_term_U(matrix_I, matrix_V)
        matrix_V = calc_term_V(matrix_I, matrix_U)
    else:
        matrix_U = np.array([(1, 1, 1), (0, np.cos(deltas[0]), np.cos(deltas[1])),
                             (0, np.sin(deltas[0]), np.sin(deltas[1]))]).T
        matrix_V = calc_term_V(matrix_I, matrix_U)
    previous_phase = deltas[1]

    error = 1.0
    iter = 1
    for iter in range(1, max_iters):
        phase, _ = calc_phase(matrix_V)
        matrix_V[:, 0] = np.ones_like(phase)
        matrix_V[:, 1] = np.cos(phase)
        matrix_V[:, 2] = -np.sin(phase)
        matrix_U = calc_term_U(matrix_I, matrix_V)

        steps, _ = calc_phase(matrix_U)
        # matrix_U[:, 0] = np.ones_like(deltas)
        matrix_U[:, 1] = np.array((0, np.cos(steps[1]), np.cos(steps[2])))
        matrix_U[:, 2] = np.array((0, np.sin(steps[1]), np.sin(steps[2])))
        matrix_V = calc_term_V(matrix_I, matrix_U)

        step = np.abs(steps[1]-steps[2])
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
