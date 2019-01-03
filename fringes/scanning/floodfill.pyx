# distutils: language = c++
import numpy as np
cimport cython
from pixel cimport Pixel
from libcpp cimport bool

cdef class FloodFill:

    cdef char[:, :] _mask
    cdef char[:, :] _visited
    cdef int _mm
    cdef int _nn
    cdef list _pixel_queue
    cdef Pixel _current_pix

    def __init__(self, (int, int) shape, Pixel start_pixel,
                 char[:,:] mask = None):
        self._mm, self._nn = shape
        self._current_pix = start_pixel
        self._pixel_queue = []

        self._visited = np.zeros((self._mm, self._nn), dtype='int8')
        if mask is None:
            self._mask = np.ones((self._mm, self._nn), dtype='int8')
        else:
            self._mask = mask

        self._start()

    cdef void _start(self):
        cdef _Lattice l_visited = _Lattice(self._visited)

        l_visited[self._current_pix] = True
        neighbors = self._current_pix.neighborhood(True)
        self._extend_pixels(neighbors)

    cdef Pixel _next_pixel(self):
        if self.empty():
            raise StopIteration
        else:
            self._current_pix = self._pixel_queue.pop(0)
            self._extend_pixels(self._current_pix.neighborhood(True))

            return self._current_pix

    cdef void _extend_pixels(self, list neighbors):
        cdef Pixel pix
        for pix in neighbors:
            if self._is_into(pix):
                    self._pixel_queue.append(pix)

    cpdef bool empty(self):
        return len(self._pixel_queue) == 0

    cdef bool _is_into(self, Pixel pix):
        if 0 <= pix._col < self._mm:
            if 0 <= pix._row < self._nn:
                if self._mask[pix._col, pix._row]:
                    if not self._visited[pix._col, pix._row]:
                        self._visited[pix._col, pix._row] = True
                        return True
        return False

    def __str__(self):
        return 'queued: {}, current: {}'.format(len(self._pixel_queue), self._current_pix)

    def __iter__(self):
        return self

    def __next__(self):
        return self._next_pixel()


cdef class _Lattice:
    cdef char[:, :] _lattice

    def __init__(self, char[:, :] array_2d):
        self._lattice = array_2d

    def __getitem__(self, Pixel item):
        return self._getitem(item)

    def __setitem__(self, Pixel key, char value):
        self._setitem(key, value)

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef char _getitem(self, Pixel item):
        return self._lattice[item._col, item._row]

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef void _setitem(self, Pixel key, char value):
        self._lattice[key._col, key._row] = value
