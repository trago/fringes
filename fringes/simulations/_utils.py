import numpy as np
from fringes.operators.basic import normalize_range


def add_speckle(phase: np.ndarray) -> np.ndarray:
    n1 = np.random.rand(*phase.shape) * np.pi * 2 - np.pi

    Ir = np.cos(n1)

    I1 = np.cos(n1 + phase)
    Ic = normalize_range((Ir + I1) ** 2,-1, 1)

    I2 = np.sin(n1 + phase)
    Is = normalize_range((Ir + I2) ** 2,-1, 1)

    return Ic + 1j * Is
