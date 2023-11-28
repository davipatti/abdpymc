import unittest

from . import simulation as sim


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
