import random
import unittest

from vision_datasets.image_object_detection.detection_as_classification_dataset import BoxAlteration


class TestBoxAlteration(unittest.TestCase):
    def test_zoom_box_out_of_range(self):
        left, t, r, b = BoxAlteration.zoom_box(10, 10, 20, 20, 100, 100, 50, 51, random.Random(0))
        assert left == 0
        assert t == 0
        assert r == 100
        assert b == 100

    def test_zoom_box_shrink_to_pt(self):
        left, t, r, b = BoxAlteration.zoom_box(10, 5, 20, 25, 100, 100, 0, 0, random.Random(0))
        assert left == 15
        assert t == 15
        assert r == 15
        assert b == 15

    def test_zoom_box_no_change(self):
        left, t, r, b = BoxAlteration.zoom_box(10, 5, 20, 25, 100, 100, 1, 1, random.Random(0))
        assert left == 10
        assert t == 5
        assert r == 20
        assert b == 25

    def test_shift_box_no_change(self):
        left, t, r, b = BoxAlteration.shift_box(10, 5, 20, 25, 100, 100, 0, 0, random.Random(0))
        assert left == 10
        assert t == 5
        assert r == 20
        assert b == 25

    def test_shift_box_rb_out(self):
        left, t, r, b = BoxAlteration.shift_box(10, 5, 20, 25, 100, 100, 50, 50, random.Random(0))
        assert left == 100
        assert t == 100
        assert r == 100
        assert b == 100

    def test_shift_box_lt_out(self):
        left, t, r, b = BoxAlteration.shift_box(10, 5, 20, 25, 100, 100, -50, -50, random.Random(0))
        assert left == 0
        assert t == 0
        assert r == 0
        assert b == 0

    def test_shift_box_rb(self):
        left, t, r, b = BoxAlteration.shift_box(10, 5, 20, 25, 100, 100, 1, 1, random.Random(0))
        assert left == 20
        assert t == 25
        assert r == 30
        assert b == 45

    def test_shift_box_lt(self):
        left, t, r, b = BoxAlteration.shift_box(10, 5, 20, 25, 100, 100, -0.1, -0.1, random.Random(0))
        assert left == 9
        assert t == 3
        assert r == 19
        assert b == 23
