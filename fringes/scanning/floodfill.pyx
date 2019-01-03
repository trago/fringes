import numpy as np
cimport cython
from pixel cimport Pixel

cdef class FloodFill:

    def __init__(self, shape, Pixel start_pixel,
                 char[:,:] mask = None):
        self._mm, self._nn = shape
        self._current_pix.col = start_pixel.col
        self._current_pix.row = start_pixel.row

        self._visited = np.zeros((self._mm, self._nn), dtype='int8')
        if mask is None:
            self._mask = np.ones((self._mm, self._nn), dtype='int8')
        else:
            self._mask = mask

        self._start()

    def __str__(self):
        return 'queued: {}, current: {}'.format(self._pixel_queue.size(),
                                                (self._current_pix.col, self._current_pix.row))

    def __iter__(self):
        return self

    def __next__(self):
        return self._next_pixel()

    def __len__(self):
        return self._pixel_queue.size()

    cpdef bool empty(self):
        return self._pixel_queue.empty()

    cdef void _start(self):
        cdef _Lattice l_visited = _Lattice(self._visited)
        cdef vector[pixel_t] neighbors = Pixel(self._current_pix.col, self._current_pix.row)._neighborhood(True)

        l_visited[self._current_pix] = True
        self._extend_pixels(neighbors)

    cdef Pixel _next_pixel(self):
        cdef pixel_t pixel
        if self.empty():
            raise StopIteration
        else:
            pixel = self._pixel_queue.front()
            self._current_pix = pixel
            self._pixel_queue.pop_front()
            obj_pixel = Pixel(pixel.col, pixel.row)
            self._extend_pixels(obj_pixel._neighborhood(True))

            return obj_pixel

    cdef void _extend_pixels(self, vector[pixel_t] neighbors):
        for n in range(8):
            if self._is_into(neighbors[n]):
                self._pixel_queue.push_back(neighbors[n])

    cdef bool _is_into(self, pixel_t pix):
        if 0 <= pix.row < self._mm:
            if 0 <= pix.col < self._nn:
                if self._mask[pix.col, pix.row]:
                    if not self._visited[pix.col, pix.row]:
                        self._visited[pix.col, pix.row] = True
                        return True
        return False

cdef class _Lattice:

    def __init__(self, char[:, :] array_2d):
        self._lattice = array_2d

    def __getitem__(self, pixel_t item):
        return self._getitem(item)

    def __setitem__(self, pixel_t key, char value):
        self._setitem(key, value)

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef char _getitem(self, pixel_t item):
        return self._lattice[item.col, item.row]

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef void _setitem(self, pixel_t key, char value):
        self._lattice[key.col, key.row] = value
