"""
Methods for processing fringe patterns with phase-shifting.

Phase-shifting fringe patterns are given like a sequence of images. Each image is a fringe pattern with a different
phase shift. This idea comes from optical interferometry.
"""

from .psi_vu import demodulate
from .psi_vu import demodulate_psi
from .psi_vu import vu_factorization
from .psi_vu import calc_phase
from .psi_vu import calc_shifts
from .psi_vu import create_matrix
