from ..scanning import FloodFill


class FloodUnwrap(FloodFill):
    def __init__(self, pp, start_pixel: Pixel = Pixel(0, 0),
                 mask: np.ndarray = None):
        super().__init__(pp.shape, start_pixel, mask)

        self._wrapped_g = pp
