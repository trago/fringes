from ..scanning import FloodFill, Pixel
from .methods import unwrap_value
from ._utils import filter_neighborhood
import numpy as np


def floodfill(pp: np.ndarray, mask: np.ndarray = None, start_at=(50, 50)):
    if mask is None:
        mask = np.ones(pp.shape, dtype=np.uint8)
    start_pixel = Pixel(start_at[0], start_at[1])
    scanner = FloodFill(pp.shape, start_pixel, mask)
    visited = np.zeros(pp.shape, dtype=np.uint8)
    up = np.zeros_like(pp)

    up[start_at[0], start_at[1]] = pp[start_pixel[0], start_pixel[1]]
    neighbors = filter_neighborhood(start_pixel.neighborhood(shuffle=True), mask, visited)
    for neighbor in neighbors:  # type: Pixel
        up[neighbor.row, neighbor.col] = unwrap_value(up[start_pixel.row, start_pixel.col],
                                                      pp[neighbor.row, neighbor.col])
    visited[start_pixel[0], start_pixel[1]] = True
    for pixel in scanner:  # type: Pixel
        neighbors = filter_neighborhood(pixel.neighborhood(), mask, visited)
        for neighbor in neighbors:  # type: Pixel
            up[neighbor.row, neighbor.col] = unwrap_value(up[pixel.row, pixel.col], pp[neighbor.row, neighbor.col])
    return up
