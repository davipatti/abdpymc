import unittest
from pathlib import Path

import numpy as np
import arviz as az

import abdpymc
from abdpymc.survival import SurvivalAnalysis, make_nan_after_last_sample


class TestSurvivalAnalysis(unittest.TestCase):
    """
    Tests for abdpymc.survival.SurvivalAnalysis.
    """

    @classmethod
    def setUpClass(cls):
        root = Path(abdpymc.__file__).parent.parent
        idata_path = Path(root, "data", "inference_data", "abd-20231202-256.nc")
        cls.idata = az.from_netcdf(idata_path)

        cohort_data_path = Path(root, "data", "cohort_data")
        cls.data = abdpymc.CombinedTiterData.from_disk(cohort_data_path)

    def test_make_instance(self):
        """
        Test can make an instance of SurvivalAnalysis.
        """
        SurvivalAnalysis(self.idata, start=0, end=10, cohort_data=self.data)

    def test_infected_dims(self):
        """
        sa.infected should be n_inds x n_int (1520, 10)
        """
        sa = SurvivalAnalysis(self.idata, start=0, end=10, cohort_data=self.data)
        self.assertEqual((1520, 9), sa.infected.shape)

    def test_exposure_dims(self):
        """
        sa.exposure should be n_inds x n_int (1520, 10)
        """
        sa = SurvivalAnalysis(self.idata, start=0, end=10, cohort_data=self.data)
        self.assertEqual((1520, 9), sa.exposure.shape)

    def test_dims_equal(self):
        """
        Test the dimensions of all arrays that should be equal.
        """
        sa = SurvivalAnalysis(self.idata, start=0, end=10, cohort_data=self.data)
        self.assertTrue(
            sa.exposure.shape
            == sa.infected.shape
            == sa.n_titer.shape
            == sa.s_titer.shape,
            f"shapes differ {sa.exposure.shape}, {sa.infected.shape}, "
            f"{sa.n_titer.shape}, {sa.s_titer.shape}",
        )

    def test_infected_max_sum_one(self):
        """
        The maximum sum of any row in infected should be 1.0.
        """
        sa = SurvivalAnalysis(self.idata, start=0, end=10, cohort_data=self.data)
        self.assertFalse((sa.infected.sum(axis=1) > 1.0).any())

    def test_exposure_min(self):
        """
        Minimum value of exposure should be 0.0.
        """
        sa = SurvivalAnalysis(self.idata, start=0, end=10, cohort_data=self.data)
        self.assertEqual(0.0, np.nanmin(sa.exposure))

    def test_exposure_max(self):
        """
        Maximum value of exposure should be 1.0.
        """
        sa = SurvivalAnalysis(self.idata, start=0, end=10, cohort_data=self.data)
        self.assertEqual(1.0, np.nanmax(sa.exposure))

    def test_initial_exposure_value_for_all_inds(self):
        """
        The initial value of exposure for all individuals should be 1.0.
        """
        sa = SurvivalAnalysis(self.idata, start=0, end=10, cohort_data=self.data)
        self.assertTrue((sa.exposure[:, 0] == 1.0).all())

    def test_exposure_decreasing(self):
        """
        All values in exposure should be monotonically non-increasing.
        """
        sa = SurvivalAnalysis(self.idata, start=0, end=10, cohort_data=self.data)
        diff = np.diff(sa.exposure, axis=1)
        diff_lt_zero = np.diff(sa.exposure, axis=1) <= 0.0
        diff_is_nan = np.isnan(diff)
        self.assertTrue(np.bitwise_or(diff_lt_zero, diff_is_nan).all())

    def test_infection_contains_nan(self):
        """
        Infection array should contain nan's past an individual's last sample.
        """
        sa = SurvivalAnalysis(self.idata, start=0, end=10, cohort_data=self.data)
        self.assertTrue(np.isnan(sa.infected).any())

    def test_exposure_nan_where_infection_nan(self):
        """
        The exposure array should have an identical pattern of nan values to the
        infection array.
        """
        sa = SurvivalAnalysis(self.idata, start=0, end=10, cohort_data=self.data)
        self.assertTrue(np.array_equal(np.isnan(sa.exposure), np.isnan(sa.infected)))


class TestMakeNanAfterLastSample(unittest.TestCase):
    """
    Tests for abdpymc.survival.make_nan_after_last_sample
    """

    def test_last_gap_negative(self):
        """
        Any values in last_gap that are negative should raise a ValueError.
        """
        arr = np.random.randn(10, 5)
        last_gap = dict(zip(np.arange(10), np.arange(-1, 9)))
        with self.assertRaisesRegex(ValueError, "values in last_gap must be positive"):
            make_nan_after_last_sample(arr, last_gap)

    def test_result(self):
        """
        Values after the last_gap should become nan.
        """
        arr = np.random.randn(3, 5)
        last_gap = {0: 3, 1: 0, 2: 10}

        expect = np.copy(arr)
        expect[0, 3 + 1 :] = np.nan
        expect[1, 0 + 1 :] = np.nan

        output = make_nan_after_last_sample(arr, last_gap)
        self.assertTrue(
            np.array_equal(expect, output, equal_nan=True),
            f"expect:\n{expect}\n\ndiffers to output:\n{output}",
        )
