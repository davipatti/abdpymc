from numbers import Number
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Antigen:
    """
    a: 50% protective titer
    b: Slope of protection curve
    """

    a: Number = 0.0
    b: Number = 1.0

    def p_protection(self, titer):
        "Probability of proctection from infection at a given titer"
        return 1 / (1 + np.exp(-self.b * (titer - self.a)))

    def is_protected(self, titer):
        "Is an individual with a given titer protected from infection?"
        return np.random.uniform() < self.p_protection(titer)

    def plot_protection(self, lo=-5, hi=5, **kwds):
        grid = np.linspace(lo, hi)
        plt.plot(grid, self.p_protection(grid), **kwds)


@dataclass
class Immunity:
    s: Antigen
    n: Antigen

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

    pcrpos: np.ndarray
    vacs: np.ndarray
    immunity: Immunity

    def __post_init__(self):
        if self.pcrpos.shape != self.vacs.shape:
            raise ValueError("vaccination and pcrpos are different shapes")
        if self.pcrpos.ndim != 1:
            raise ValueError("vaccination and pcrpos should be 1 dimensional")
        self.n_gaps = len(self.pcrpos)

    def infection_responses(
        self, s_init: Number, n_init: Number, lam0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        for t in range(self.n_gaps):
            is_exposed = np.random.uniform() < lam0[t]

            is_infected = (
                not self.immunity.is_protected(s_titer=s[t], n_titer=n[t])
                if is_exposed
                else False
            )


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

    immunity: Immunity

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
