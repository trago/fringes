from typing import Dict, Union, Tuple, List

import numpy as np

from ..operators.basic import normalize_range


def peaks(mm: int, nn: int) -> np.ndarray:
    y, x = np.ogrid[-3.0:3.0:mm * 1j, -3.0:3.0:nn * 1j]

    p = 1 - x / 2.0 + x ** 5 + y ** 3
    p *= np.exp(-x ** 2 - y ** 2)

    return p


def gaussian(sigma: float = 10.0, shape: Tuple[int, int] = (512, 512)) -> np.ndarray:
    mm, nn = shape
    y, x = np.ogrid[-mm / 2:mm / 2:mm * 1j, -nn / 2:nn / 2:nn * 1j]

    g_values = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return g_values


def ramp(M: int, N: int, ku: int = 32, kv: int = 0) -> np.ndarray:
    y, x = np.ogrid[-M / 2.0:M / 2.0:M * 1j, -N / 2.0:N / 2.0:N * 1j]
    u = 2 * np.pi * ku / N
    v = 2 * np.pi * kv / M
    return u * x + v * y


def add_speckle(phase: np.ndarray) -> np.ndarray:
    n1 = np.random.rand(*phase.shape) * np.pi * 2 - np.pi

    Ir = np.cos(n1)

    I1 = np.cos(n1 + phase)
    Ic = (Ir + I1) ** 2

    I2 = np.sin(n1 + phase)
    Is = (Ir + I2) ** 2

    return Ic + 1j * Is


def parabola(M, N, cm=-1, cn=-1):
    cm = M / 2 if cm < 0 else cm
    cn = N / 2 if cn < 0 else cn

    y, x = np.ogrid[0:M, 0:N]

    return (x - cn) ** 2 + (y - cm) ** 2


def interferogram(shape, dc: Union[str, Dict[str, Union[float, Tuple[float, float]]]] = 'constant',
                  phase: Union[str, Dict[str, Union[float, Tuple[int, int]]]] = 'peaks',
                  magn: Union[str, Dict[str, float]] = 'constant',
                  noise: Dict[str, Tuple[float, float]] = 'clean',
                  phase_shift=0.0, normalize=False) -> np.ndarray:
    mm, nn = shape

    if isinstance(dc, str):
        dc = {dc: 1.0}
    if isinstance(phase, str):
        phase = {phase: 1.0}
    if isinstance(magn, str):
        magn = {magn: 1.0}
    if isinstance(noise, str):
        noise = {noise: 0.0}

    arg_type = [key for key in phase.keys()][0]
    if arg_type == 'peaks':
        phase = peaks(mm, nn) * phase['peaks']
    elif arg_type == 'ramp':
        ku, kv = phase['ramp']
        phase = ramp(mm, nn, ku, kv)
    elif arg_type == 'parabola':
        phase = parabola(mm, nn) * phase['parabola']
    phase += phase_shift

    arg_type = [key for key in dc.keys()][0]
    if arg_type == 'constant':
        dc = dc['constant']
    elif arg_type == 'gaussian':
        dc = gaussian(dc['gaussian'][0], (mm, nn)) * dc['gaussian'][1]

    arg_type = [key for key in magn.keys()][0]
    if arg_type == 'constant':
        magn = magn['constant']
    elif arg_type == 'gaussian':
        magn = gaussian(magn['gaussian'][0], (mm, nn)) * magn['gaussian'][1]

    arg_type = [key for key in noise.keys()][0]
    img = np.zeros((mm, nn), dtype=float)
    if arg_type == 'normal':
        img = dc + magn * np.cos(phase)
        img += np.random.randn(mm, nn) * np.sqrt(noise['normal'][1]) + noise['normal'][0]
    elif arg_type == 'speckle':
        img = dc + magn * add_speckle(phase).real
    elif arg_type == 'clean':
        img = dc + magn * np.cos(phase)

    if normalize:
        img = normalize_range(img, -1, 1)

    return img


def interferogram_psi(steps: Union[Tuple[int, float], List[float], np.ndarray] = (5, np.pi / 2),
                      shape=(512, 512), dc: Union[str, Dict[str, Union[float, Tuple[float, float]]]] = 'constant',
                      phase: Union[str, Dict[str, Union[float, Tuple[int, int]]]] = 'peaks',
                      magn: Union[str, Dict[str, float]] = 'constant',
                      noise: Dict[str, Tuple[float, float]] = 'clean', normalize=True) -> List[np.ndarray]:
    if isinstance(steps, tuple):
        steps = [n * steps[1] for n in range(steps[0])]

    images = []
    for step in steps:
        img = interferogram(shape, dc, phase, magn, noise, step, normalize)
        images.append(img)

    return images
