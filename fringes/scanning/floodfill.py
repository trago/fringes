from typing import Tuple, List
from .pixel import Pixel

import numpy as np


class FloodFill:

    def __init__(self, shape: Tuple[int, int], start_pixel: Pixel = Pixel(0, 0),
                 mask: np.ndarray = None):
        self._mask: np.ndarray = mask
        self._mm, self._nn = shape
        self._current_pix: Pixel = start_pixel
        self._pixel_queue: List[Pixel] = []

        self._visited = np.zeros((self._mm, self._nn), dtype=np.bool)
        if self._mask is None:
            self._mask = np.ones((self._mm, self._nn), dtype=np.bool)

        self._start()

    def _start(self):
        l_visited = _Lattice(self._visited)

        l_visited[self._current_pix] = True
        neighbors = self._current_pix.neighborhood(shuffle=True)
        self._extend_pixels(neighbors)

    def next(self):
        if self.empty():
            raise StopIteration
        else:
            self._current_pix = self._pixel_queue.pop(0)
            self._extend_pixels(self._current_pix.neighborhood(shuffle=True))

            return self._current_pix

    def _extend_pixels(self, neighbors):
        for pix in neighbors:
            if self._is_into(pix):
                    self._pixel_queue.append(pix)

    def empty(self):
        return len(self._pixel_queue) == 0

    def _is_into(self, pix):
        if 0 <= pix.col < self._mm:
            if 0 <= pix.row < self._nn:
                if self._mask[pix.col, pix.row]:
                    if not self._visited[pix.col, pix.row]:
                        self._visited[pix.col, pix.row] = True
                        return True
        return False

    def __str__(self):
        return 'queued: {}, current: {}'.format(len(self._pixel_queue), self._current_pix)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


class _Lattice:

    def __init__(self, array_2d: np.ndarray):
        self._lattice = array_2d

    def __getitem__(self, item):
        return self._lattice[item[0], item[1]]

    def __setitem__(self, key, value):
        self._lattice[key[0], key[1]] = value

