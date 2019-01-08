import unittest
import numpy as np

from skimage.viewer import CollectionViewer
from skimage.io import imread

from fringes.operators.basic import normalize_range, wrap
from fringes.simulations import wavefront
from fringes.unwrap import floodfill_unwrap, find_inconsistencies, erode_mask, dilating_unwrap, erode_unwrap
from fringes.unwrap.methods import unwrap_value


class unwrap(unittest.TestCase):

    def test_unwrap_value(self):
        """
        `unwrap_value` receives two parameters and returns the value that unwraps the second argument.

        The wrapping operator is the round operator.
        """
        v1 = 12.4
        v2 = 1.2
        v2 = unwrap_value(v1, v2)

        self.assertTrue(abs(v2 - v1) < 0.5)

    def test_floodfill(self):
        phase = wavefront((256, 512), {'parabola': 0.0005}, noise={'normal': (0, 0.5)}, normalize=True)
        up = floodfill_unwrap(phase, start_at=(128, 128))

        # viewer = CollectionViewer([normalize_range(up), normalize_range(phase)])
        # viewer.show()

    def test_find_inconsistencies_00(self):
        phase = wavefront((256, 512), {'peaks': 15}, noise={'normal': (0, 0.5)}, normalize=True)
        mask = find_inconsistencies(phase)
        up = floodfill_unwrap(phase, mask, start_at=(128, 256))
        viewer = CollectionViewer([normalize_range(mask),
                                   normalize_range(up),
                                   normalize_range(phase)])
        viewer.show()

    def test_find_inconsistencies_01(self):
        phase = imread('../data/wFaseEstatica.png')
        phase = normalize_range(phase, -0.5, 0.5)

        mask = find_inconsistencies(phase)
        up = floodfill_unwrap(phase, mask, start_at=(128, 256))
        viewer = CollectionViewer([normalize_range(mask),
                                   normalize_range(up),
                                   normalize_range(phase)])
        viewer.show()

    def test_find_inconsistencies_02(self):
        phase = imread('../data/dificult.png', True)
        phase = normalize_range(phase, -0.5, 0.5)*2.005
        phase = wrap(phase)

        mask = find_inconsistencies(phase)
        up = floodfill_unwrap(phase, mask, start_at=(128, 128))
        viewer = CollectionViewer([normalize_range(mask),
                                   normalize_range(up),
                                   normalize_range(phase)])
        viewer.show()

    def test_find_inconsistencies_03(self):
        phase = imread('../data/membrane.png', True)
        mask = imread('../data/membrane_mask.png', True)
        mask = normalize_range(mask, 0, 1).astype(np.uint8)
        phase = normalize_range(phase, -.5, .5)
        phase = wrap(phase)

        inconsistencies = find_inconsistencies(phase, mask)
        inconsistencies_mask = inconsistencies.copy()
        inconsistencies_mask[mask == 0] = 0
        viewer = CollectionViewer([normalize_range(inconsistencies),
                                   normalize_range(inconsistencies_mask),
                                   normalize_range(mask),
                                   normalize_range(phase)])
        viewer.show()

    def test_erode_mask_00(self):
        phase = imread('../data/wFaseEstatica.png', True)
        phase = normalize_range(phase, -0.5, 0.5)
        phase = wrap(phase)

        mask = find_inconsistencies(phase)
        dilated_mask = erode_mask(mask)
        viewer = CollectionViewer([normalize_range(mask),
                                   normalize_range(dilated_mask),
                                   normalize_range(phase)])
        viewer.show()

    def test_erode_mask_01(self):
        phase = imread('../data/wFaseEstatica.png', True)
        phase = normalize_range(phase, -0.5, 0.5)
        phase = wrap(phase)

        mask = find_inconsistencies(phase)
        dilated_mask = erode_mask(mask, 3)
        viewer = CollectionViewer([normalize_range(mask),
                                   normalize_range(dilated_mask),
                                   normalize_range(phase)])
        viewer.show()

    def test_dilate_unwrap_01(self):
        phase = imread('../data/wFaseEstatica.png', True)
        phase = normalize_range(phase, -0.5, 0.5)
        phase = wrap(phase)

        up, mask_result = dilating_unwrap(phase, start_at=(256, 256), max_iters=15)
        viewer = CollectionViewer([normalize_range(up),
                                   normalize_range(mask_result),
                                   normalize_range(phase)])
        viewer.show()

    def test_dilate_unwrap_02(self):
        phase = wavefront((256, 512), {'peaks': 15}, noise={'normal': (0, 0.43)}, normalize=True)
        phase = wrap(phase)

        up, mask_result = dilating_unwrap(phase, start_at=(128, 256), max_iters=15)
        mask = find_inconsistencies(phase)
        viewer = CollectionViewer([normalize_range(up),
                                   normalize_range(mask_result),
                                   normalize_range(mask),
                                   normalize_range(phase)])
        viewer.show()

    def test_dilate_unwrap_03(self):
        phase = imread('../data/dificult.png', True)
        phase = normalize_range(phase, -0.5, 0.5)*2.005
        phase = wrap(phase)

        up, mask_result = dilating_unwrap(phase, start_at=(128, 128), max_iters=100)
        mask = find_inconsistencies(phase)
        viewer = CollectionViewer([normalize_range(up),
                                   normalize_range(mask_result),
                                   normalize_range(mask),
                                   normalize_range(phase)])
        viewer.show()

    def test_dilate_unwrap_04(self):
        phase = imread('../data/membrane.png', True)
        mask = imread('../data/membrane_mask.png', True)
        mask = normalize_range(mask, 0, 1).astype(np.uint8)
        phase = normalize_range(phase, -.5, .5)
        phase = wrap(phase)

        up, inconsistencies_dilated = dilating_unwrap(phase, mask, start_at=(225, 350), max_iters=100)
        inconsistencies = find_inconsistencies(phase)
        viewer = CollectionViewer([normalize_range(up),
                                   normalize_range(inconsistencies_dilated),
                                   normalize_range(inconsistencies),
                                   normalize_range(phase)])
        viewer.show()

    def test_erode_unwrap_01(self):
        phase = imread('../data/dificult.png', True)
        phase = normalize_range(phase, -0.5, 0.5)*2.005
        phase = wrap(phase)

        up, mask_result = dilating_unwrap(phase, start_at=(128, 128), max_iters=100)
        up, new_mask = erode_unwrap(phase, up, mask_result)

        print(new_mask.min(), new_mask.max())

        viewer = CollectionViewer([normalize_range(up),
                                   normalize_range(mask_result),
                                   normalize_range(new_mask),
                                   normalize_range(phase)])
        viewer.show()

    def test_erode_unwrap_02(self):
        phase = imread('../data/wFaseEstatica.png', True)
        phase = normalize_range(phase, -0.5, 0.5)

        up, mask_result = dilating_unwrap(phase, start_at=(228, 128), max_iters=100)
        up, new_mask = erode_unwrap(phase, up, mask_result)

        viewer = CollectionViewer([normalize_range(up),
                                   normalize_range(mask_result),
                                   normalize_range(new_mask),
                                   normalize_range(phase)])
        viewer.show()

    def test_erode_unwrap_03(self):
        phase = imread('../data/membrane.png', True)
        mask = imread('../data/membrane_mask.png', True)
        phase = normalize_range(phase, -0.5, 0.5)
        mask = normalize_range(mask, 0, 1).astype(np.uint8)
        phase = wrap(phase)

        up1, inconsistencies = dilating_unwrap(phase, mask, start_at=(225, 350), max_iters=100)
        up2, mask_result = erode_unwrap(phase, up1, inconsistencies, mask, 0)
        viewer = CollectionViewer([normalize_range(up1),
                                   normalize_range(up2),
                                   normalize_range(mask_result),
                                   normalize_range(mask),
                                   normalize_range(phase)])
        viewer.show()
