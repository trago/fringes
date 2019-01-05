import numpy as np


def filter_neighborhood(neighbors: list, mask: np.ndarray, visited: np.ndarray):
    valid_neighbors = []
    mm, nn = mask.shape
    for pixel in neighbors:  # type: Pixel
        if 0 <= pixel.row < mm and 0 <= pixel.col < nn:
            if mask[pixel.row, pixel.col]:
                if not visited[pixel.row, pixel.col]:
                    visited[pixel.row, pixel.col] = True
                    valid_neighbors.append(pixel)
                    # print(pixel)
    return valid_neighbors
