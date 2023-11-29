from collections import namedtuple
from numbers import Number
from pydantic import (
    BaseModel,
    NonNegativeFloat,
    PositiveFloat,
    FiniteFloat,
    field_validator,
)
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

InfectionResponses = namedtuple(
    "InfectionResponses", ("infections", "s_response", "n_response")
)


class Response(BaseModel):
    """
    a: 50% protective titer
    b: Slope of protection curve


    """

    a: FiniteFloat = 0.0
    b: PositiveFloat = 1.0
    perm_rise: PositiveFloat = 2.0
    temp_rise: PositiveFloat = 1.5
    temp_wane: PositiveFloat = 0.95
    temp: NonNegativeFloat = 0.0
    init: FiniteFloat = -2.0

    @field_validator("temp_wane")
    @classmethod
    def wane_must_be_between_0_1(cls, value):
        if not 0 <= value <= 1:
            raise ValueError("waning parameter must be between 0-1")
        return value

    def p_protection(self, titer: float) -> float:
        "Probability of proctection from infection at a given titer"
        return 1 / (1 + np.exp(-self.b * (titer - self.a)))

    def is_protected(self, titer: float) -> float:
        "Is an individual with a given titer protected from infection?"
        return np.random.uniform() < self.p_protection(titer)

    def plot_protection(self, lo=-5, hi=5, **kwds):
        grid = np.linspace(lo, hi)
        plt.plot(grid, self.p_protection(grid), **kwds)

    def next_temp_response(self, prev: float, is_infected: bool) -> float:
        """
        Calculate the next temporary response given a previous temporary response.

        Args:
            prev: The previous temporary response.
            is_infected: Whether or not this individual was infected.
        """
        return prev * self.temp_wane + is_infected * self.temp_rise

    def perm_response(self, infections: np.array) -> float:
        """
        Calculate a permanent response, given an array of infections.
        """
        return self.perm_response if infections.any() else 0.0


@dataclass
class Responses:
    s: Response
    n: Response

    def is_protected(self, s_titer: Number, n_titer: Number) -> bool:
        "Is an individual protected by either an S or N response?"
        return self.s.is_protected(s_titer) or self.n.is_protected(n_titer)

    def plot_protection(self, lo=-5, hi=5, **kwds):
        self.s.plot_protection(lo=lo, hi=hi, label="S", **kwds)
        self.n.plot_protection(lo=lo, hi=hi, label="N", **kwds)


@dataclass
class Individual:
    """
    {pcrpos,vacs}: Both (n_gaps,) binary arrays indicating when individuals had a PCR+
        result or were vaccinated.
    """

    pcrpos: np.array
    vacs: np.array
    responses: Responses

    def __post_init__(self):
        if self.pcrpos.shape != self.vacs.shape:
            raise ValueError("vaccination and pcrpos are different shapes")
        if self.pcrpos.ndim != 1:
            raise ValueError("vaccination and pcrpos should be 1 dimensional")
        self.n_gaps = len(self.pcrpos)

    def infection_responses(
        self, s_init: Number, n_init: Number, lam0: np.array
    ) -> tuple[np.array, np.array, np.array]:
        """
        Simulate infections and responses.

        Args:
            s_init: Starting S titer.
            n_init: Starting N titer.
            lam0: Per-time gap infection probability (baseline infection rate).
        """
        if lam0.ndim != 1:
            raise ValueError("lam0 should be 1D")

        if len(lam0) != self.n_gaps:
            raise ValueError("must have single infection rate for each time gap")

        s = np.empty(self.n_gaps)
        n = np.empty(self.n_gaps)
        infections = np.zeros(self.n_gaps)

        s[0] = s_init
        n[0] = n_init

        s_temp = 0
        n_temp = 0

        for t in range(self.n_gaps):
            is_exposed = np.random.uniform() < lam0[t]
            is_infected = (
                not self.responses.is_protected(s_titer=s[t], n_titer=n[t])
                if is_exposed
                else False
            )

            if is_infected:
                infections[t] = 1.0

            s_temp = self.responses.s.next_temp_response(
                s_temp, is_infected=is_infected
            )
            n_temp = self.responses.n.next_temp_response(
                n_temp, is_infected=is_infected
            )

            s_perm = self.responses.s.perm_response(infections)
            n_perm = self.responses.n.perm_response(infections)

            s[t] = s_init + s_temp + s_perm
            n[t] = n_init + n_temp + n_perm

        return InfectionResponses(infections=infections, s_response=s, n_response=n)


@dataclass
class Cohort:
    """
    vacs_path: Path to file containing an (n_inds, n_gaps) binary array where 1 indicates
        an individual was vaccinated in that time gap.
    pcrpos_path: Path to file containing an (n_inds, n_gaps) binary array where 1
        indicates an individual had a positive PCR test in that time gap.
    {s,n}_mu: Mean of individuals initial S/N titer.
    {s,n}_sd: Standard deviation of individuals initial S/N titer.

    Inferred attributes:
        n_inds: Number of individuals.
        n_gaps: Number of time gaps.
    """

    random_seed: int

    vacs_path: str
    pcrpos_path: str

    responses: Responses

    s_mu: Number = -2
    n_mu: Number = -2
    s_sd: Number = 1
    n_sd: Number = 1

    def __post_init__(self):
        vacs = np.loadtxt(self.vacs_path)
        pcrpos = np.loadtxt(self.pcrpos_path)
        if vacs.shape != pcrpos.shape:
            raise ValueError("vaccination and PCR+ data are different shapes")

        np.random.seed(self.random_seed)

        self.n_inds, self.n_gaps = vacs.shape

        self.s_titer = np.empty(vacs.shape)
        self.n_titer = np.empty(vacs.shape)

        self.s_titer[:, 0] = np.random.randn(self.n_inds) * self.s_sd + self.s_mu
        self.n_titer[:, 0] = np.random.randn(self.n_inds) * self.n_sd + self.n_mu
