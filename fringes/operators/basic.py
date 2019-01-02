import numpy as np
from numba import jit, vectorize, float32, float64


@jit(cache=True, nopython=True, nogil=True)
def grad(p):
    M, N = p.shape
    dx = np.zeros(p.shape)
    dy = np.zeros(p.shape)

    for n in range(N):
        for m in range(M):
            if m - 1 >= 0:
                dy[m, n] = p[m, n] - p[m - 1, n]
            elif m + 1 < M:
                dy[m, n] = p[m + 1, n] - p[m, n]
            else:
                dy[m, n] = p[m, n]
            if n - 1 >= 0:
                dx[m, n] = p[m, n] - p[m, n - 1]
            elif n + 1 < N:
                dx[m, n] = p[m, n + 1] - p[m, n]
            else:
                dx[m, n] = p[m, n]

    return dx, dy


@vectorize([float64(float64),
            float32(float32)], cache=True, nopython=True, nogil=True)
def wrap(p):
    return p - round(p)


def normalize_range(g: np.ndarray, min_v=0, max_v=1):
    a = g.min()
    b = g.max()

    return (max_v-min_v)*(g-a)/(b-a) + min_v
