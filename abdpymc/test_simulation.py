from pathlib import Path
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

    def test_temp_rise_i_must_be_positive(self):
        with self.assertRaisesRegex(ValidationError, "temp_rise_i"):
            sim.Response(temp_rise_i=-1)

    def test_perm_rise_must_be_positive(self):
        with self.assertRaisesRegex(ValidationError, "perm_rise"):
            sim.Response(perm_rise=-1)

    def test_next_temp_response(self):
        ag = sim.Response(temp_wane=0.95)
        temp_response = ag.next_temp_response(prev=1.0, is_infected=False)
        self.assertEqual(1.0 * 0.95, temp_response)

    def test_next_temp_response_with_infection(self):
        ag = sim.Response(temp_wane=0.95, temp_rise_i=1.8)
        temp_response = ag.next_temp_response(prev=1.0, is_infected=True)
        self.assertEqual(1.0 * 0.95 + 1.8, temp_response)

    def test_cant_pass_unknown_arg(self):
        with self.assertRaisesRegex(ValidationError, "Extra inputs are not permitted"):
            sim.Response(xyz=123)

    def test_perm_response_incl_vaccination(self):
        ag = sim.Response(temp_wane=0.95, temp_rise_i=1.8, perm_rise=2.2)
        perm_response = ag.perm_response(
            infections=np.zeros(5), vaccinations=np.array([0, 0, 1, 0, 0])
        )
        self.assertEqual(2.2, perm_response)


class TestResponses(unittest.TestCase):
    def test_default_responses(self):
        """
        Should be able to make an instance without providing any values.
        """
        sim.Responses()

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
    def test_responses_default_values(self):
        """
        Should be able to instantiate without passing Responses.
        """
        sim.Individual(pcrpos=np.array([0, 0, 0, 0, 0]), vacs=np.array([0, 0, 1, 0, 0]))

    def test_vacs_pcrpos(self):
        """
        Test vacs and pcrpos attributes are passed correctly.
        """
        ind = sim.Individual(
            pcrpos=np.array([0, 0, 0, 0, 0]), vacs=np.array([0, 0, 1, 0, 0])
        )
        self.assertTrue((ind.vacs == np.array([0, 0, 1, 0, 0])).all())
        self.assertTrue((ind.pcrpos == np.array([0, 0, 0, 0, 0])).all())

    def test_cant_pass_vacs_pcrpos_diff_shape(self):
        """
        Passing vacs and pcrpos that are different shapes should raise a ValueError.
        """
        with self.assertRaisesRegex(
            ValueError, "vaccination and pcrpos are different shapes"
        ):
            sim.Individual(pcrpos=np.array([0, 0, 1]), vacs=np.array([1, 0, 0, 0]))

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
            )

    def test_infection_responses_returns_3tuple(self):
        """
        Infection responses should return a 3-namedtuple containing np.ndarrays
        """
        ind = sim.Individual(pcrpos=np.array([0, 0, 1]), vacs=np.array([0, 1, 0]))

        output = ind.infection_responses(lam0=np.array([0.1, 0.1, 0.1]))

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
            pcrpos=np.array([0, 0, 0]),
            vacs=np.array([0, 0, 0]),
            responses=sim.Responses(
                s=sim.Response(init=0.123), n=sim.Response(init=0.456)
            ),
        )

        output = ind.infection_responses(lam0=np.array(np.zeros(3)))

        self.assertTrue((output.s_response == 0.123).all())
        self.assertTrue((output.n_response == 0.456).all())

    def test_pcrpos_get_recognised_as_infections(self):
        """
        A PCR+ should cause a 1 in the infections output.
        """
        ind = sim.Individual(
            pcrpos=np.array([0, 0, 1, 0, 0]), vacs=np.array([0, 0, 0, 0, 0])
        )
        output = ind.infection_responses(lam0=np.array(np.zeros(5)))
        self.assertTrue((output.infections == np.array([0, 0, 1, 0, 0])).all())

    def test_pcrpos_response_s(self):
        """
        Test S responses are correct after a PCR+.
        """
        ind = sim.Individual(
            pcrpos=np.array([0, 0, 1, 0, 0]),
            vacs=np.array([0, 0, 0, 0, 0]),
            responses=sim.Responses(
                s=sim.Response(temp_rise_i=0.3, perm_rise=0.34, temp_wane=0.94, init=-1)
            ),
        )

        output = ind.infection_responses(lam0=np.array(np.zeros(5)))

        # Before the PCR+
        self.assertTrue((np.array([-1, -1]) == output.s_response[:1]).all())

        # At the point of the PCR+
        self.assertAlmostEqual(-1 + 0.3 + 0.34, output.s_response[2])

        # One time step after PCR+
        self.assertAlmostEqual(-1 + 0.3 * 0.94 + 0.34, output.s_response[3])

    def test_pcrpos_response_n(self):
        """
        Test N responses are correct after a PCR+.
        """
        ind = sim.Individual(
            pcrpos=np.array([0, 0, 1, 0, 0]),
            vacs=np.array([0, 0, 0, 0, 0]),
            responses=sim.Responses(
                n=sim.Response(temp_rise_i=0.89, perm_rise=2.34, temp_wane=0.87),
            ),
        )

        output = ind.infection_responses(lam0=np.array(np.zeros(5)))

        # Before the PCR+
        self.assertTrue((np.array([-2, -2]) == output.n_response[:1]).all())

        # At the point of the PCR+
        self.assertAlmostEqual(-2 + 0.89 + 2.34, output.n_response[2])

        # One time step after PCR+
        self.assertAlmostEqual(-2 + 0.89 * 0.87 + 2.34, output.n_response[3])

    def test_vaccination_effect_on_s(self):
        """
        Vaccinations should increase the S response.
        """
        ind = sim.Individual(
            pcrpos=np.array([0, 0, 0, 0, 0]),
            vacs=np.array([0, 0, 1, 0, 0]),
            responses=sim.Responses(
                s=sim.Response(temp_rise_v=0.3, perm_rise=0.34, temp_wane=0.94, init=-1)
            ),
        )

        output = ind.infection_responses(lam0=np.array(np.zeros(5)))

        # Before the vaccination
        self.assertTrue((np.array([-1, -1]) == output.s_response[:1]).all())

        # At the point of the vaccination
        self.assertAlmostEqual(-1 + 0.3 + 0.34, output.s_response[2])

        # One time step after vaccination
        self.assertAlmostEqual(-1 + 0.3 * 0.94 + 0.34, output.s_response[3])

    def test_vaccination_and_pcrpos_effect_on_s(self):
        """
        Vaccinations should increase the S response.
        """
        ind = sim.Individual(
            pcrpos=np.array([0, 0, 0, 1, 0]),
            vacs=np.array([0, 0, 1, 0, 0]),
            responses=sim.Responses(
                s=sim.Response(
                    temp_rise_i=0.21,
                    temp_rise_v=0.3,
                    perm_rise=0.34,
                    temp_wane=0.94,
                    init=-1,
                )
            ),
        )

        output = ind.infection_responses(lam0=np.array(np.zeros(5)))

        # Before vaccination
        self.assertTrue((np.array([-1, -1]) == output.s_response[:1]).all())

        # At the point of the vaccination
        self.assertAlmostEqual(-1 + 0.3 + 0.34, output.s_response[2])

        # One time step after vaccination, gets infected
        self.assertAlmostEqual(-1 + 0.3 * 0.94 + 0.34 + 0.21, output.s_response[3])

    def test_vaccination_and_pcrpos_effect_on_n(self):
        """
        Only PCR+ should impact N response.
        """
        ind = sim.Individual(
            pcrpos=np.array([0, 0, 0, 1, 0]),
            vacs=np.array([0, 0, 1, 0, 0]),
            responses=sim.Responses(
                n=sim.Response(
                    temp_rise_v=0.0, temp_rise_i=0.89, perm_rise=2.34, temp_wane=0.87
                ),
            ),
        )

        output = ind.infection_responses(lam0=np.array(np.zeros(5)))

        # Before PCR+ all values should be initial
        self.assertTrue((np.array([-2, -2, -2]) == output.n_response[:3]).all())

        # One time step after vaccination, gets infected
        self.assertAlmostEqual(-2 + 0.89 + 2.34, output.n_response[3])


class TestCohort(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cohort_data_path = Path(Path(sim.__file__).parent.parent, "data", "cohort_data")
        cls.cohort = sim.Cohort(random_seed=42, cohort_data_path=cohort_data_path)

    def test_fresh_cohort_lacks_n_titer(self):
        """
        Cohort instances shouldn't have s_titer or n_titer attributes when they are
        instantiated.
        """
        self.assertFalse(hasattr(self.cohort, "n_titer"))

    def test_fresh_cohort_lacks_s_titer(self):
        """
        Cohort instances shouldn't have s_titer or n_titer attributes when they are
        instantiated.
        """
        self.assertFalse(hasattr(self.cohort, "s_titer"))

    def test_simulate_responses_s_titer_shape(self):
        """
        Calling .simulate_responses should attach an s_titer attribute that has the same
        shape as self.vacs.
        """
        lam0 = np.repeat(0.04, self.cohort.n_gaps)
        self.cohort.simulate_responses(lam0=lam0)
        self.assertEqual(self.cohort.true.vacs.shape, self.cohort.s_titer.shape)

    def test_simulate_responses_n_titer_shape(self):
        """
        Calling .simulate_responses should attach an n_titer attribute that has the same
        shape as self.vacs.
        """
        lam0 = np.repeat(0.04, self.cohort.n_gaps)
        self.cohort.simulate_responses(lam0=lam0)
        self.assertEqual(self.cohort.true.vacs.shape, self.cohort.n_titer.shape)

    def test_simulate_responses_infections_shape(self):
        """
        Calling .simulate_responses should attach an infections attribute that has the
        same shape as self.vacs.
        """
        lam0 = np.repeat(0.04, self.cohort.n_gaps)
        self.cohort.simulate_responses(lam0=lam0)
        self.assertEqual(self.cohort.true.vacs.shape, self.cohort.infections.shape)
