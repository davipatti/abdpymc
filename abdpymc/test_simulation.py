import unittest

import numpy as np
from pydantic import ValidationError

import abdpymc.simulation as sim


class TestResponse(unittest.TestCase):
    def test_p_protection_high_titer(self):
        """
        At a very high titer, probability of protection should be very close to 1.
        """
        p = sim.Response(a=0, b=1).p_protection(100)
        self.assertAlmostEqual(1.0, p)

    def test_p_protection_different_a(self):
        """
        Probability of protection at a given titer should be lower with a higher a (and
        positive b).
        """
        self.assertLess(
            sim.Response(a=1, b=1).p_protection(0),
            sim.Response(a=0, b=1).p_protection(0),
        )

    def test_p_protection_different_titer(self):
        """
        Probability of protection should be higher at a higher titer (when b is
        positive).
        """
        ag = sim.Response(a=0, b=1)
        self.assertGreater(ag.p_protection(1), ag.p_protection(0))

    def test_temp_wane_must_be_between_0_1(self):
        with self.assertRaisesRegex(ValidationError, "temp_wane"):
            sim.Response(temp_wane=1.5)

        with self.assertRaisesRegex(ValidationError, "temp_wane"):
            sim.Response(temp_wane=-0.5)

    def test_temp_rise_must_be_positive(self):
        with self.assertRaisesRegex(ValidationError, "temp_rise"):
            sim.Response(temp_rise=-1)

    def test_perm_rise_must_be_positive(self):
        with self.assertRaisesRegex(ValidationError, "perm_rise"):
            sim.Response(perm_rise=-1)

    def test_next_temp_response(self):
        ag = sim.Response(temp_wane=0.95)
        temp_response = ag.next_temp_response(prev=1.0, is_infected=False)
        self.assertEqual(1.0 * 0.95, temp_response)

    def test_next_temp_response_with_infection(self):
        ag = sim.Response(temp_wane=0.95, temp_rise=1.8)
        temp_response = ag.next_temp_response(prev=1.0, is_infected=True)
        self.assertEqual(1.0 * 0.95 + 1.8, temp_response)


class TestResponses(unittest.TestCase):
    def test_protected_n_alone(self):
        """
        Individual should be protected if N titer alone is high enough.
        """
        imm = sim.Responses(s=sim.Response(a=100, b=1), n=sim.Response(a=-100, b=1))
        self.assertTrue(imm.is_protected(s_titer=0, n_titer=0))

    def test_protected_s_alone(self):
        """
        Individual should be protected if S titer alone is high enough.
        """
        imm = sim.Responses(s=sim.Response(a=-100, b=1), n=sim.Response(a=100, b=1))
        self.assertTrue(imm.is_protected(s_titer=0, n_titer=0))

    def test_not_protected(self):
        """
        Individual should not be protected if both S an N titer are very low.
        """
        imm = sim.Responses(s=sim.Response(a=0, b=1), n=sim.Response(a=0, b=1))
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
                responses=sim.Responses(
                    s=sim.Response(a=0, b=1), n=sim.Response(a=0, b=1)
                ),
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
                responses=sim.Responses(
                    s=sim.Response(a=0, b=1), n=sim.Response(a=0, b=1)
                ),
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
                responses=sim.Responses(
                    s=sim.Response(a=0, b=1), n=sim.Response(a=0, b=1)
                ),
            )

    def test_infection_responses_returns_3tuple(self):
        """
        Infection responses should return a 3-namedtuple containing np.ndarrays
        """
        ind = sim.Individual(
            pcrpos=np.array([0, 0, 1]),
            vacs=np.array([0, 1, 0]),
            responses=sim.Responses(s=sim.Response(a=0, b=1), n=sim.Response(a=0, b=1)),
        )

        output = ind.infection_responses(
            s_init=0, n_init=0, lam0=np.array([0.1, 0.1, 0.1])
        )

        self.assertIsInstance(output, sim.InfectionResponses)
        self.assertEqual(3, len(output))
        self.assertIsInstance(output.s_response, np.ndarray)
        self.assertIsInstance(output.n_response, np.ndarray)
        self.assertIsInstance(output.infections, np.ndarray)

        for arr in output:
            self.assertEqual(3, len(arr))

    def test_no_responses_if_lam0_0(self):
        """
        If the baseline infection rate is always 0 then responses shouldn't change from
        initial responses.
        """
        ind = sim.Individual(
            pcrpos=np.array([0, 0, 1]),
            vacs=np.array([0, 1, 0]),
            responses=sim.Responses(s=sim.Response(a=0, b=1), n=sim.Response(a=0, b=1)),
        )

        output = ind.infection_responses(
            s_init=0.123, n_init=0.456, lam0=np.array(np.zeros(3))
        )

        self.assertTrue((output.s_response == 0.123).all())
        self.assertTrue((output.n_response == 0.456).all())
