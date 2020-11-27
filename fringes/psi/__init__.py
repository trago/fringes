"""
Methods for processing fringe patterns with phase-shifting.

Phase-shifting fringe patterns are given like a sequence of images. Each image is a fringe pattern with a different
phase shift. This idea comes from optical interferometry.
"""

from .psi_vu import demodulate
from .psi_vu import demodulate_psi
from .psi_vu import vu_factorization
from .psi_vu import calc_phase
from .psi_vu import create_matrix
from .psi_vu import calc_term_U, calc_term_V
from .psi_2frames import demodulate as demodulate_2frames
from .psi_2frames import phase_2steps
from .psi_2frames import demodulate_2steps
