from numba import jit, njit
from numba import types as tp


@jit(cache=True, nopython=True)
def unwrap_value(v1, v2):
    wrap_diff = round(v2 - v1)
    return v2 - wrap_diff


def unwrap(pp):
    pass
