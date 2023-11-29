import unittest

import numpy as np

import abdpymc.simulation as sim


class TestAntigen(unittest.TestCase):
    def test_p_protection_high_titer(self):
        """
        At a very high titer, probability of protection should be very close to 1.
        """
        p = sim.Antigen(a=0, b=1).p_protection(100)
        self.assertAlmostEqual(1.0, p)

    def test_p_protection_different_a(self):
        """
        Probability of protection at a given titer should be lower with a higher a (and
        positive b).
        """
        self.assertLess(
            sim.Antigen(a=1, b=1).p_protection(0), sim.Antigen(a=0, b=1).p_protection(0)
        )

    def test_p_protection_different_titer(self):
        """
        Probability of protection should be higher at a higher titer (when b is
        positive).
        """
        ag = sim.Antigen(a=0, b=1)
        self.assertGreater(ag.p_protection(1), ag.p_protection(0))


class TestImmunity(unittest.TestCase):
    def test_protected_n_alone(self):
        """
        Individual should be protected if N titer alone is high enough.
        """
        imm = sim.Immunity(s=sim.Antigen(a=100, b=1), n=sim.Antigen(a=-100, b=1))
        self.assertTrue(imm.is_protected(s_titer=0, n_titer=0))

    def test_protected_s_alone(self):
        """
        Individual should be protected if S titer alone is high enough.
        """
        imm = sim.Immunity(s=sim.Antigen(a=-100, b=1), n=sim.Antigen(a=100, b=1))
        self.assertTrue(imm.is_protected(s_titer=0, n_titer=0))

    def test_not_protected(self):
        """
        Individual should not be protected if both S an N titer are very low.
        """
        imm = sim.Immunity(s=sim.Antigen(a=0, b=1), n=sim.Antigen(a=0, b=1))
        self.assertFalse(imm.is_protected(s_titer=-100, n_titer=-100))


class TestIndividual(unittest.TestCase):
    def test_cant_pass_vacs_pcrpos_diff_shape(self):
        """
        Passing vacs and pcrpos that are different shapes should raise a ValueError.
        """
        with self.assertRaisesRegex(
            ValueError, "vaccination and pcrpos are different shapes"
        ):
            sim.Individual(
                pcrpos=np.array([0, 0, 1]),
                vacs=np.array([1, 0, 0, 0]),
                immunity=sim.Immunity(s=sim.Antigen(a=0, b=1), n=sim.Antigen(a=0, b=1)),
            )

    def test_vacs_must_be_1d(self):
        """
        Vacs must be 1D.
        """
        with self.assertRaisesRegex(
            ValueError, "vaccination and pcrpos should be 1 dimensional"
        ):
            sim.Individual(
                pcrpos=np.array([[0, 0, 1], [0, 1, 0]]),
                vacs=np.array([[0, 0, 1], [0, 1, 0]]),
                immunity=sim.Immunity(s=sim.Antigen(a=0, b=1), n=sim.Antigen(a=0, b=1)),
            )

    def test_pcrpos_must_be_1d(self):
        """
        pcrpos must be 1D.
        """
        with self.assertRaisesRegex(
            ValueError, "vaccination and pcrpos should be 1 dimensional"
        ):
            sim.Individual(
                pcrpos=np.array([[0, 0, 1], [0, 1, 0]]),
                vacs=np.array([[0, 0, 1], [0, 1, 0]]),
                immunity=sim.Immunity(s=sim.Antigen(a=0, b=1), n=sim.Antigen(a=0, b=1)),
            )
