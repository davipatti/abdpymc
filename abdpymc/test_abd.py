#!/usr/bin/env python3

import logging
import unittest
from pathlib import Path

import numpy as np
import pytensor.tensor as at
import pymc as pm

import abdpymc as abd


class TestMaskFutureInfections(unittest.TestCase):
    """Tests for abd.mask_future_infection"""

    def test_no_prior_infections(self):
        """Infection should be allowed if there are no prior infections"""
        self.assertEqual(abd.mask_future_infection(1, 0, 0, 0).eval(), 1)

    def test_no_prior_infection_no_current_infection(self):
        """Check infections don't appear from nowhere."""
        self.assertEqual(abd.mask_future_infection(0, 0, 0, 0).eval(), 0)

    def test_current_value_is_returned(self):
        """
        Whatever is in the "i0" slot should be returned, if there are no
        prior infections.
        """
        self.assertEqual(abd.mask_future_infection(10, 0, 0, 0).eval(), 10)

    def test_infection_in_slot_m3_masks(self):
        """An infection 3 slots before should mask a current infection."""
        self.assertEqual(abd.mask_future_infection(1, 1, 0, 0).eval(), 0)

    def test_infection_in_slot_m2_masks(self):
        """An infection 2 slots before should mask a current infection."""
        self.assertEqual(abd.mask_future_infection(1, 0, 1, 0).eval(), 0)

    def test_infection_in_slot_m1_masks(self):
        """An infection 1 slots before should mask a current infection."""
        self.assertEqual(abd.mask_future_infection(1, 0, 0, 1).eval(), 0)

    def test_infection_in_slot_m3_doesnt_create_infection(self):
        """
        An infection 3 slots before current should not produce an
        infection.
        """
        self.assertEqual(abd.mask_future_infection(0, 1, 0, 0).eval(), 0)

    def test_infection_in_slot_m2_doesnt_create_infection(self):
        """
        An infection 2 slots before current should not produce an
        infection.
        """
        self.assertEqual(abd.mask_future_infection(0, 0, 1, 0).eval(), 0)

    def test_infection_in_slot_m1_doesnt_create_infection(self):
        """
        An infection 1 slots before current should not produce an
        infection.
        """
        self.assertEqual(abd.mask_future_infection(0, 0, 0, 1).eval(), 0)


class TestAllITitersData(unittest.TestCase):
    """
    Tests for abd.AllITitersData
    """

    @classmethod
    def setUp(cls):
        """
        Build an instance of CombinedAllITitersData for testing. This object has 's' and
        'n' attributes which are instances of AllITitersData.
        """
        root = Path(abd.__file__).parent.parent
        directory = root.joinpath(Path("data", "cohort_data"))
        cls.cat = abd.CombinedTiterData.from_disk(directory)

    def test_idx_gap_shape(self):
        """
        Index to extract values of a from (n_gaps, n_inds) matrix should be same length
        as the data.
        """
        self.assertEqual(len(self.cat.n.idx_gap), len(self.cat.n.df))

    def test_idx_ind_shape(self):
        """
        Index to extract values of a from (n_gaps, n_inds) matrix should be same length
        as the data.
        """
        self.assertEqual(len(self.cat.n.idx_ind), len(self.cat.n.df))


class TestMaskInfectionsWithin3Months(unittest.TestCase):
    """
    Tests for functions that prevent infections occurring immediately after one another.
    When an infection occurs, an infection is not allowed to occur in the following
    three time chunks.
    """

    def test_mask_infections_3_months_shape(self):
        """
        The output shape should match the input shape.
        """
        input = at.as_tensor(
            [
                [1, 1, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
            ]
        )
        output = abd.mask_three_gaps(input).eval()
        self.assertEqual((5, 3), output.shape)

    def test_mask_infections_within_3_months(self):
        """
        An infection should not be possible 3 months after another infection.
        """
        input = at.as_tensor_variable(
            [
                [1, 1, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
            ]
        )
        output = abd.mask_three_gaps(input).eval()
        expect = at.as_tensor_variable(
            [
                [1, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
            ]
        )
        nrows, ncols = input.shape.eval()
        for i in range(nrows):
            for j in range(ncols):
                with self.subTest(i=i, j=j):
                    self.assertEqual(expect[i, j].eval(), output[i, j])


class TestMaskMultipleInfections(unittest.TestCase):
    """
    Tests for functionality around constraining indiviudals to only be allowed a single
    infection per variant.

    mask_multiple_infections takes a tensor like:

    [
        [0, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1]
    ]

    And turns any 1 that is preceded by another 1 in a column to 0.
    """

    @classmethod
    def setUp(cls):
        """
        An array for testing.
        """
        np.random.seed(42)
        cls.n_gap = 29
        cls.n_ind = 50
        cls.arr = at.as_tensor_variable(
            np.random.randint(0, 2, size=(cls.n_gap, cls.n_ind))
        )

    def test_returns_same_shape(self):
        """
        Should return the same shape array that is passed to it.
        """
        out = abd.mask_multiple_infections(self.arr)
        self.assertEqual((29, 50), tuple(out.shape.eval()))

    def test_col_max_1(self):
        """
        The maximum sum of any column in the returned array should be 1.
        """
        out = abd.mask_multiple_infections(self.arr)
        colsums = out.sum(axis=0).eval()
        self.assertEqual(1, max(colsums))

    def test_returns_0_or_1(self):
        """
        Only 0 or 1 are allowed values in the returned array.
        """
        out = abd.mask_multiple_infections(self.arr)
        self.assertEqual({0, 1}, set(out.eval().ravel()))

    def test_exact_value(self):
        """
        Test the correct value is returned.
        """
        input = np.array([[0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 1]])
        expect = np.array([[0, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]])
        out = abd.mask_multiple_infections(input).eval()
        self.assertTrue(np.all(np.equal(expect, out)))


class TestMaskMultipleInfectionsChunks(unittest.TestCase):
    """
    Tests to check functionality where masking multiple infections is split into
    different chunks. A single infection is allowed within each chunk of the array.
    The first infection in a chunk takes precedence.

    E.g. if this array were split into two chunks:

    [
        [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],  Chunk 1
        #------------------------------------------------------
        [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],  Chunk 2
        [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
    ]

    Then this should be the product:

    [
        [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  Chunk 1
        #------------------------------------------------------
        [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],  Chunk 2
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    """

    @classmethod
    def setUp(cls):
        """
        An array for testing.
        """
        np.random.seed(42)
        cls.n_gap = 29
        cls.n_ind = 50
        cls.arr = at.as_tensor_variable(
            np.random.randint(0, 2, size=(cls.n_gap, cls.n_ind))
        )

    def test_returns_same_shape(self):
        """
        Should return the same shape array that is passed to it.
        """
        out = abd.mask_multiple_infections_2_chunks(self.arr, split=18)
        self.assertEqual((29, 50), tuple(out.shape.eval()))

    def test_col_max_2(self):
        """
        The maximum sum of any column in the returned array should be 2.
        """
        out = abd.mask_multiple_infections_2_chunks(self.arr, split=18)
        colsums = out.sum(axis=0).eval()
        self.assertEqual(2, max(colsums))

    def test_returns_0_or_1(self):
        """
        Only 0 or 1 are allowed values in the returned array.
        """
        out = abd.mask_multiple_infections_2_chunks(self.arr, split=18)
        self.assertEqual({0, 1}, set(out.eval().ravel()))

    def test_exact_value_single_split(self):
        """
        Test the correct value is returned for the single split case.
        """
        input = at.as_tensor_variable(
            [
                [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
                [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
                # ---------------------------------------------
                [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
            ]
        )
        expect = [
            [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # --------------------------------------------
            [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        out = abd.mask_multiple_infections_2_chunks(input, split=4).eval()
        self.assertTrue(np.all(np.equal(expect, out)))

    def test_exact_value_double_split(self):
        """
        Test the correct value is returned for the double split case.
        """
        input = at.as_tensor_variable(
            [
                [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
                [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                [0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                # ---------------------------------------------
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
                # ---------------------------------------------
                [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
                [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0],
                [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            ]
        )
        expect = [
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            # ---------------------------------------------
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # ---------------------------------------------
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        out = abd.mask_multiple_infections_3_chunks(input, split0=4, split1=12).eval()
        self.assertTrue(np.all(np.equal(expect, out)))


class TestIncorporatePCRPos(unittest.TestCase):
    def test_cases(self):
        """
        Each column in these arrays test a different case:

        Column:
            1. PCR+ after infection should take precedence
            2. PCR+ before infection should take precedence
            3. No PCR+ -> i_raw should be unchanged
            4. PCR+ without i_raw -> should match PCR+ row
            5. No PCR+ or infection -> all zeros
            6. Multiple PCR+ -> all PCR+ are retained.
        """
        pcr_pos = np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        i_raw = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        expect = np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        out = abd.incorporate_pcrpos(i_raw=i_raw, pcrpos=pcr_pos).eval()
        self.assertTrue(np.all(np.equal(expect, out)), f"output:\n{np.array(out)}")


class TestTwoTimeChunks(unittest.TestCase):
    def test_split_create_chunks_correct_length(self):
        """
        - 10 total time gaps
        - Split at 4 should create a chunk of length 4 and a chunk of length 6.
        """
        pcrpos = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )

        # split at 4, making one length 4 chunk and one length 6 chunk
        ttc = abd.TwoTimeChunks(split=4, pcrpos=at.as_tensor(pcrpos))

        # Test that the sizes of the PCR pos arrays are as expected.
        self.assertEqual(4, ttc.pcrpos_0.shape.eval()[0])
        self.assertEqual(6, ttc.pcrpos_1.shape.eval()[0])

    def test_pcrpos_chunk1_dont_mask_chunk2(self):
        """
        A PCR+ in the first time chunk shouldn't mask inferred infections in the second
        time chunk.

        Also, multiple PCR+ in one chunk should appear in the output.
        """
        pcrpos = [
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1],  # Chunk 1
            # -------- Split ---
            [0, 0],  # Chunk 2
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 0],
        ]
        i_raw = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],  # Chunk 1
            # -------- Split ---
            [0, 0],  # Chunk 2
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1],
            [0, 0],
        ]

        ttc = abd.TwoTimeChunks(split=4, pcrpos=at.as_tensor(pcrpos))
        output = ttc.incorporate_pcrpos(i_raw=at.as_tensor(i_raw)).eval()

        expect = [
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1],  # Chunk 1
            # -------- Split ---
            [0, 0],  # Chunk 2
            [0, 1],
            [1, 0],
            [0, 0],
            [0, 1],
            [0, 0],
        ]

        self.assertTrue(np.all(np.equal(expect, output)))

    def test_multiple_infections_survive_incorporate_pcrpos(self):
        """
        Test that TwoTimeChunks.incorporate_pcrpos
        """
        pcrpos = [
            [0],
            [0],
            [0],
            [1],
            # ------ Split ---
            [0],
            [1],
            [0],
            [0],
            [1],
            [0],
        ]
        i_raw = [
            [0],
            [0],
            [0],
            [0],
            # ------ Split ---
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
        ]

        ttc = abd.TwoTimeChunks(split=4, pcrpos=at.as_tensor(pcrpos))

        expect = [
            [0],
            [0],
            [0],
            [1],
            # ------ Split ---
            [0],
            [1],
            [0],
            [0],
            [1],
            [0],
        ]

        with pm.Model(coords=dict(gap=range(10), ind=range(2))):
            output = ttc.incorporate_pcrpos(i_raw=at.as_tensor(i_raw)).eval()

        self.assertTrue(
            np.all(np.equal(expect, output)), f"output:\n{np.array(output)}"
        )


class TestTwoTimeChunksConstrainInfections(unittest.TestCase):
    """
    MultipleTimeChunks is the base class for TestTwoTimeChunks.

    MultipleTimeChunks.constrain_infections combines mask_multiple_infections and
    incorporate_pcrpos.
    """

    def test_constrain_infections(self):
        """
        Testing the behaviour of TwoTimeChunks.constrain_infections.

        Columns:
            1: Two PCR+ separated by 3 gaps - expect both in output.
            2: Two PCR+ separated by 2 gaps - expect first in output.
            3: Three PCR+ separated each by 2 gaps - expect first and third.
        """
        pcrpos = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            # ------ Split ---
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]
        i_raw = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            # ------ Split ---
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        ttc = abd.TwoTimeChunks(split=4, pcrpos=at.as_tensor(pcrpos))

        expect = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            # ------ Split ---
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 1],
        ]

        with pm.Model(coords=dict(gap=range(10), ind=range(3))):
            output = ttc.constrain_infections(i_raw=at.as_tensor(i_raw)).eval()

        self.assertTrue(
            np.all(np.equal(expect, output)), f"output:\n{np.array(output)}"
        )


class TestInvLogistic(unittest.TestCase):
    """
    Tests for abdpymc.logistic and abdpymc.invlogistic.
    """

    def test_invlogistic_is_inverse_of_logistic(self):
        """
        invlogistic should be the inverse of logistic.
        """
        kwds = dict(a=-0.123, b=1.357, d=7.78)
        self.assertEqual(3, abd.logistic(abd.invlogistic(3, **kwds), **kwds))


class CombinedTiterDataPath:
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(
            Path(abd.__file__).parent.parent, "data", "test_data", "cohort_data"
        )


class TestCombinedTiterData(CombinedTiterDataPath, unittest.TestCase):
    """
    Tests for abdpymc.CombinedTiterData.
    """

    def test_coords_inds(self):
        """
        `ind` coord should be 0, ..., n_inds - 1
        """
        data = abd.CombinedTiterData.from_disk(self.directory)
        self.assertEqual(list(range(10)), list(data.coords["ind"]))

    def test_coords_gaps(self):
        """
        `gap` coord should be 0, ..., n_gaps - 1
        """
        data = abd.CombinedTiterData.from_disk(self.directory)
        self.assertEqual(list(range(26)), list(data.coords["gap"]))


class TestModel(CombinedTiterDataPath, unittest.TestCase):
    """
    Tests for abdpymc.model.
    """

    def test_test_data(self):
        """
        Test the test data is as expected.
        """
        data = abd.CombinedTiterData.from_disk(self.directory)
        self.assertEqual(10, data.n_inds)
        self.assertEqual(26, data.n_gaps)
        self.assertEqual((10, 26), data.vacs.shape)
        self.assertEqual((10, 26), data.pcrpos.shape)

    def test_indexes_no_splits(self):
        """
        Test that the only indexes present are gap and ind.
        """
        data = abd.CombinedTiterData.from_disk(self.directory)
        with abd.model(data=data):
            idata = pm.sample_prior_predictive(samples=1)
        self.assertEqual({"chain", "draw", "gap", "ind"}, set(idata.prior.indexes))

    def test_indexes_split_delta(self):
        """
        Test that the only indexes present are gap and ind.
        """
        data = abd.CombinedTiterData.from_disk(self.directory)
        splits = data.calculate_splits(delta=True, omicron=False)
        with abd.model(data=data, splits=splits):
            idata = pm.sample_prior_predictive(samples=1)
        self.assertEqual({"chain", "draw", "gap", "ind"}, set(idata.prior.indexes))

    def test_indexes_split_omicron(self):
        """
        Test that the only indexes present are gap and ind.
        """
        data = abd.CombinedTiterData.from_disk(self.directory)
        splits = data.calculate_splits(delta=False, omicron=True)
        with abd.model(data=data, splits=splits):
            idata = pm.sample_prior_predictive(samples=1)
        self.assertEqual({"chain", "draw", "gap", "ind"}, set(idata.prior.indexes))

    def test_indexes_split_delta_omicron(self):
        """
        Test that the only indexes present are gap and ind.
        """
        data = abd.CombinedTiterData.from_disk(self.directory)
        splits = data.calculate_splits(delta=True, omicron=True)
        with abd.model(data=data, splits=splits):
            idata = pm.sample_prior_predictive(samples=1)
        self.assertEqual({"chain", "draw", "gap", "ind"}, set(idata.prior.indexes))


class TestComputeChunkedCumProbability(unittest.TestCase):
    def test_cum_p_never_exceeds_one(self):
        """
        Cummulative probability should not exceed 1.
        """
        p = np.random.uniform(low=0.5, high=0.8, size=20)
        cum_p = abd.compute_chunked_cum_p(p)
        self.assertEqual(1.0, max(cum_p))

    def test_cum_p_never_exceeds_one_two_chunks(self):
        """
        Cummulative probability should not exceed 1.
        """
        p = np.random.uniform(low=0.7, high=0.8, size=20)
        cum_p = abd.compute_chunked_cum_p(p, splits=(10,))
        self.assertEqual(1.0, max(cum_p[:10]), "value in first chunk exceeds 1.0")
        self.assertEqual(1.0, max(cum_p[10:]), "value in second chunk exceeds 1.0")

    def test_cum_p_two_chunks_probabilities_reset(self):
        """
        Probabilities should go back to being less than one at the start of the second
        chunk.
        """
        p = np.random.uniform(low=0.7, high=0.8, size=20)
        cum_p = abd.compute_chunked_cum_p(p, splits=(10,))
        self.assertLess(cum_p[10], 1.0)

    def test_raises_error_if_p_not_1d(self):
        """
        Should raise a ValueError if p is not 1D.
        """
        with self.assertRaisesRegex(ValueError, "p is not 1D"):
            p = np.random.uniform(0, 1, size=(10, 10))
            abd.compute_chunked_cum_p(p)


class TestTempResponse(unittest.TestCase):
    """
    Tests for abd.temp_response
    """

    def test_temp_response_shape(self):
        """
        The output shape should match the input shape.
        """
        n_gaps = 10
        n_inds = 5
        exposure = at.as_tensor_variable(np.random.randint(0, 2, size=(n_gaps, n_inds)))
        response = abd.temp_response(exposure, n_inds=n_inds, temp=1.0, rho=0.9).eval()
        self.assertEqual(tuple(exposure.shape.eval()), (n_gaps, n_inds))
        self.assertEqual(response.shape, (n_gaps, n_inds))

    def test_temp_response_values(self):
        """
        Test the correct values are returned.
        """
        exposure = at.as_tensor_variable(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )
        response = abd.temp_response(exposure, n_inds=3, temp=1.0, rho=0.5).eval()
        expected_response = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.25, 0.5, 1.0],
                [1.125, 0.25, 0.5],
                [0.5625, 1.125, 0.25],
            ]
        )
        np.testing.assert_almost_equal(response, expected_response)

    def test_temp_response_values_more_gaps_more_individuals(self):
        """
        Like above, but bigger array.
        """
        exposure = at.as_tensor_variable(
            [
                [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            ]
        )
        response = abd.temp_response(exposure, n_inds=11, temp=1.0, rho=0.5).eval()
        expected_response = np.array(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
                [0.25, 0.5, 1.0, 0.25, 0.5, 1.0, 0.25, 0.5, 1.0, 0.25, 0.5],
                [1.125, 0.25, 0.5, 1.125, 0.25, 0.5, 1.125, 0.25, 0.5, 1.125, 0.25],
                [
                    0.5625,
                    1.125,
                    0.25,
                    0.5625,
                    1.125,
                    0.25,
                    0.5625,
                    1.125,
                    0.25,
                    0.5625,
                    1.125,
                ],
                [
                    0.28125,
                    0.5625,
                    1.125,
                    0.28125,
                    0.5625,
                    1.125,
                    0.28125,
                    0.5625,
                    1.125,
                    0.28125,
                    0.5625,
                ],
                [
                    1.140625,
                    0.28125,
                    0.5625,
                    1.140625,
                    0.28125,
                    0.5625,
                    1.140625,
                    0.28125,
                    0.5625,
                    1.140625,
                    0.28125,
                ],
                [
                    0.5703125,
                    1.140625,
                    0.28125,
                    0.5703125,
                    1.140625,
                    0.28125,
                    0.5703125,
                    1.140625,
                    0.28125,
                    0.5703125,
                    1.140625,
                ],
                [
                    0.28515625,
                    0.5703125,
                    1.140625,
                    0.28515625,
                    0.5703125,
                    1.140625,
                    0.28515625,
                    0.5703125,
                    1.140625,
                    0.28515625,
                    0.5703125,
                ],
                [
                    1.142578125,
                    0.28515625,
                    0.5703125,
                    1.142578125,
                    0.28515625,
                    0.5703125,
                    1.142578125,
                    0.28515625,
                    0.5703125,
                    1.142578125,
                    0.28515625,
                ],
                [
                    0.5712890625,
                    1.142578125,
                    0.28515625,
                    0.5712890625,
                    1.142578125,
                    0.28515625,
                    0.5712890625,
                    1.142578125,
                    0.28515625,
                    0.5712890625,
                    1.142578125,
                ],
                [
                    0.28564453125,
                    0.5712890625,
                    1.142578125,
                    0.28564453125,
                    0.5712890625,
                    1.142578125,
                    0.28564453125,
                    0.5712890625,
                    1.142578125,
                    0.28564453125,
                    0.5712890625,
                ],
                [
                    1.142822265625,
                    0.28564453125,
                    0.5712890625,
                    1.142822265625,
                    0.28564453125,
                    0.5712890625,
                    1.142822265625,
                    0.28564453125,
                    0.5712890625,
                    1.142822265625,
                    0.28564453125,
                ],
                [
                    0.5714111328125,
                    1.142822265625,
                    0.28564453125,
                    0.5714111328125,
                    1.142822265625,
                    0.28564453125,
                    0.5714111328125,
                    1.142822265625,
                    0.28564453125,
                    0.5714111328125,
                    1.142822265625,
                ],
                [
                    0.28570556640625,
                    0.5714111328125,
                    1.142822265625,
                    0.28570556640625,
                    0.5714111328125,
                    1.142822265625,
                    0.28570556640625,
                    0.5714111328125,
                    1.142822265625,
                    0.28570556640625,
                    0.5714111328125,
                ],
            ]
        )
        np.testing.assert_almost_equal(response, expected_response)

    def test_temp_response_no_exposure(self):
        """
        Test the response when there is no exposure.
        """
        exposure = at.as_tensor_variable(np.zeros((10, 5)))
        response = abd.temp_response(exposure, n_inds=5, temp=1.0, rho=0.9).eval()
        expected_response = np.zeros((10, 5))
        np.testing.assert_almost_equal(response, expected_response)

    def test_compare_to_scan_implementation(self):
        """
        Test that abd.temp_response gives the same result as abd._temp_response_scan for 5 different
        combinations of random parameters that are passed to these functions.
        """
        np.random.seed(42)
        for _ in range(5):
            n_gaps = np.random.randint(1, 20)
            n_inds = np.random.randint(1, 10)
            exposure = at.as_tensor_variable(
                np.random.randint(0, 2, size=(n_gaps, n_inds))
            )
            temp = np.random.uniform(0.1, 2.0)
            rho = np.random.uniform(0.1, 0.9)

            kwds = dict(exposure=exposure, n_inds=n_inds, temp=temp, rho=rho)

            with self.subTest(
                n_gaps=n_gaps, n_inds=n_inds, temp=temp, rho=rho, exposure=exposure
            ):

                response = abd.temp_response(**kwds).eval()
                response_scan = abd.abd._temp_response_scan(**kwds).eval()

                np.testing.assert_almost_equal(response, response_scan)


if __name__ == "__main__":
    # Stop pymc reporting things like which variables are going to be sampled
    logging.getLogger("pymc").setLevel(logging.ERROR)

    unittest.main()
