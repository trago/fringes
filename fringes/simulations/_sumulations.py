from typing import Dict, Union, List

from ._utils import add_speckle
from .functions import *
from ..operators.basic import normalize_range


def interferogram(shape: Tuple[int, int],
                  dc: Union[str, Dict[str, Union[float, Tuple[float, float]]]] = 'constant',
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

    if phase.get('peaks', None) is not None:
        phase = peaks(mm, nn) * phase['peaks']
    elif phase.get('ramp', None) is not None:
        ku, kv = phase['ramp']
        phase = ramp(mm, nn, ku, kv)
    elif phase.get('parabola', None) is not None:
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


def wavefront(shape: Tuple[int, int],
              phase: Union[str, Dict[str, Union[float, Tuple[int, int]]]] = 'peaks',
              noise: Dict[str, Tuple[float, float]] = 'clean',
              normalize=False) -> np.ndarray:
    mm, nn = shape

    if isinstance(phase, str):
        phase = {phase: 1.0}
    if isinstance(noise, str):
        noise = {noise: 0.0}

    if phase.get('peaks', None) is not None:
        phase = peaks(mm, nn) * phase['peaks']
    elif phase.get('ramp', None) is not None:
        ku, kv = phase['ramp']
        phase = ramp(mm, nn, ku, kv)
    elif phase.get('parabola', None) is not None:
        phase = parabola(mm, nn) * phase['parabola']

    img = np.zeros((mm, nn), dtype=float)
    if noise.get('normal', None) is not None:
        phase_noise = np.random.randn(mm, nn) * np.sqrt(noise['normal'][1]) + noise['normal'][0]
        img_cc = np.cos(phase + phase_noise)
        img_ss = np.sin(phase + phase_noise)
        img = np.arctan2(img_ss, img_cc)
    elif noise.get('speckle', None) is not None:
        speckle = add_speckle(phase)
        img = np.arctan2(speckle.imag, speckle.real)
    elif noise.get('clean', None) is not None:
        img_cc = np.cos(phase)
        img_ss = np.sin(phase)
        img = np.arctan2(img_ss, img_cc)

    if normalize:
        img /= 2*np.pi

    return img
