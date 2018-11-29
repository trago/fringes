import numpy as np


def vu_factorization(matrix_I: np.ndarray, error_accuracy: float = 1e-3, max_iters: int = 20, verbose=False,
                     verbose_step = 5):
    matrix_I += 1000
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

def calc_factor_U(matrix_I: np.ndarray, factor_V):
    aux_Ainv = np.linalg.inv(factor_V.T @ factor_V)
    factor_U = aux_Ainv @ factor_V.T @ matrix_I

    return factor_U.T


def calc_factor_V(matrix_I: np.ndarray, factor_U):
    aux_Binv = np.linalg.inv(factor_U.T @ factor_U)
    factor_V = matrix_I @ factor_U @ aux_Binv

    return factor_V

def create_matrix(image_list):
    shape = image_list[0].shape
    N = len(image_list)
    M = shape[0]*shape[1]
    matrix_images = np.zeros((M,N))

    for k, img in enumerate(image_list):
        if shape != img.shape:
            raise TypeError('Images in the list must have the same dimension')
        matrix_images[:, k] = img.reshape(1, shape[0] * shape[1])

    return matrix_images

def calc_phase(matrix_V):
    return np.arctan2(-matrix_V[:,2], matrix_V[:,1])

def calc_shifts(matrix_U):
    return np.arctan2(matrix_U[:,2], matrix_U[:,1])

def calc_magnitude(matrix_psi):
    return np.absolute(matrix_psi[:,1] + 1j*matrix_psi[:,2])

def print_iter_info(iter, error, error_tol, verbose_step):
    if iter % verbose_step == 0:
        notif = '\tIteration {0:04}: Objective E: {1:10.3e} Current E:{2:10.3e}'. \
            format(iter, error_tol, error)
        print(notif)

def print_report_info(iter, error, error_tol):
    print()
    if error < error_tol:
        print('Process finished with relative error = {0:10.4e}, iterations = {1}'.
              format(error, iter))
    else:
        print('Process finished by reaching the maximum number of iterations with error = {0:10.4e}'.
              format(error))


def demodulate(image_list, error_accuracy=1e-3, max_iters=20, verbose=False, verbose_step = 5):
    matrix_form = create_matrix(image_list)
    factor_V, factor_U = vu_factorization(matrix_form, error_accuracy, max_iters, verbose, verbose_step)

    return calc_phase(factor_V).reshape(image_list[0].shape)

def demodulate_psi(image_list, phase_step, initial_step=0.0):
    N = len(image_list)
    if isinstance(phase_step, list):
        deltas = np.array(phase_step) - initial_step
    elif isinstance(phase_step, np.ndarray):
        deltas = phase_step
    else:
        deltas = np.array([(n*phase_step - initial_step) for n in range(N)])

    image_matrix = create_matrix(image_list)

    s1_ones = np.ones(N)
    factor_U = np.vstack([s1_ones, np.cos(deltas), np.sin(deltas)]).T
    factor_V = calc_factor_V(image_matrix, factor_U)

    return calc_phase(factor_V).reshape(image_list[0].shape)