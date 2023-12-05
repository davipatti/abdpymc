import unittest

from abdpymc.plotting import overhanging_slice


class TestOverHangingSlice(unittest.TestCase):

    """
    Tests for abdpymc.plotting.overhanging_slice
    """

    def test_case_a(self):
        self.assertEqual((5,), overhanging_slice(range(10), center=5, pad=0))

    def test_case_b(self):
        self.assertEqual((4, 5, 6), overhanging_slice(range(10), center=5, pad=1))

    def test_case_c(self):
        self.assertEqual(
            (None, None, 0, 1, 2), overhanging_slice(range(10), center=0, pad=2)
        )

    def test_case_d(self):
        self.assertEqual(
            (None, 0, 1, 2, 3), overhanging_slice(range(10), center=1, pad=2)
        )

    def test_case_e(self):
        self.assertEqual(
            (5, 6, 7, 8, 9, None, None), overhanging_slice(range(10), center=8, pad=3)
        )
