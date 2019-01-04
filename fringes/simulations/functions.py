from typing import Tuple

import numpy as np


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


def parabola(M, N, cm=-1, cn=-1):
    cm = M / 2 if cm < 0 else cm
    cn = N / 2 if cn < 0 else cn

    y, x = np.ogrid[0:M, 0:N]

    return (x - cn) ** 2 + (y - cm) ** 2
