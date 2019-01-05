import unittest
from skimage.viewer import ImageViewer, CollectionViewer
from fringes.unwrap.methods import unwrap_value
from fringes.unwrap import floodfill, find_inconsistencies
from fringes.simulations import wavefront
from fringes.operators.basic import normalize_range


class unwrap(unittest.TestCase):

    def test_unwrap_value(self):
        v1 = 12.4
        v2 = 1.2
        v2 = unwrap_value(v1, v2)

        self.assertTrue(abs(v2 - v1) < 0.5)

    def test_floodfill(self):
        phase = wavefront((256, 512), {'parabola': 0.0005}, noise={'normal':(0, 0.5)}, normalize=True)
        up = floodfill(phase, start_at=(128, 256))

        viewer = CollectionViewer([normalize_range(up), normalize_range(phase)])
        viewer.show()

    def test_find_inconsistencies(self):
        phase = wavefront((256, 512), {'parabola': 0.0005}, noise={'normal': (0, 0.4)}, normalize=True)
        mask = find_inconsistencies(phase)
        up = floodfill(phase, mask, start_at=(128, 256))
        viewer = CollectionViewer([normalize_range(mask),
                                   normalize_range(up),
                                   normalize_range(phase)])
        viewer.show()
