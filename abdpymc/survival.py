from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Literal, Callable

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


@dataclass
class SurvivalAnalysis:
    """
    Conduct a survival analysis on an antibody dynamics run.

    Time:
        'start' and 'end' delimit the first and last time intervals used in the analysis.
        There are actually (end - start - 1) intervals in the survival analysis because
        infection risk during interval t is modelled as a function of titer at time t-1.
        (There is no risk modelled in the first time slice.)

                  `start`              `end`
                     |                   |
        titer:      [t0, t1, t2, t3, t4, t5]
                        ╲   ╲   ╲   ╲   ╲
        infection:  [i0, i1, i2, i3, i4, i5]


    Args:
        idata: InferenceData object
        start: First time slice of the analysis.
        end: Last time slice of the analysis.
        data: CombinedAllITititersData containing raw cohort data.

    Attributes:
        n_int: Number of intervals.
        intervals: end - start array containing [0, ..., n_int - 2].
        t0: Pandas month period corresponding to the first interval in the infected and
            exposure arrays (i.e. one month after `start`).
        infected: (n_ind, n_int) array of mean infection probabilities processed
            such that the maximum cumulative infection probability for any individual is
            one.
        exposure: (n_ind, n_int)
        {s,n}_titer: (n_ind, n_int)
        sn_titer: (n_ind, n_int, 2)

    """

    idata: "arviz.InferenceData"
    start: int
    end: int
    cohort_data: "abdpymc.CombinedTiterData"

    def __post_init__(self):
        self.post = az.extract(self.idata)
        self.post_mean = self.post.mean(dim="sample")

        self.intervals = np.arange(self.end - self.start - 1)
        self.n_int = len(self.intervals)
        self.t0 = self.cohort_data.t0 + self.start + 1

        # Get the raw infection probabilities. Transpose because inference data arrays
        # are all (gap x ind), but have implemented the survival analysis code to work on
        # (ind x gap) arrays
        infected_raw = self.post_mean["i"].values.T

        # Remove all infection probabilities that occur after the last sample was taken
        infected_raw = make_nan_after_last_sample(
            infected_raw, last_gap=self.cohort_data.last_gap
        )

        # Extract infection probability for time gaps that are used in this survival
        # analysis.
        self.infected = infected_raw[:, self.start + 1 : self.end]

        # Using value_while_cumsum_below_threshold here means that the total infection
        # probability for each individual (row) never exceeds 1.0. I.e. at most each
        # individual is allowed a single infection.
        self.infected = np.array(
            [
                list(value_while_cumsum_below_threshold(row, threshold=1.0))
                for row in self.infected
            ]
        )

        # Exposure in the first time gap for an individual is 1.0. In subsequent gaps it
        # decreases by the cumulative sum of the infection probability from preceding
        # gaps.
        self.exposure = np.ones_like(self.infected)
        self.exposure[:, 1:] -= self.infected.cumsum(axis=1)[:, :-1]

        # Make sure that the pattern of missing data (np.nan) is the same in exposure as
        # in infected. Because exposure is calculated from the cumulative infection
        # probability that is lagged by one time gap (see above), nan's appear one time
        # step too late in the exposure array without this step.
        self.exposure[np.isnan(self.infected)] = np.nan

        # Extract antibody titer inferences. These arrays are the same size as `infected`
        # and `exposure`, but offset by one time gap such that the month `m` in the
        # titer array is aligned with month `m+1` in the exposure and infection arrays.
        self.s_titer = self.post_mean["ab_s_mu"].values.T[:, self.start : self.end - 1]
        self.n_titer = self.post_mean["ab_n_mu"].values.T[:, self.start : self.end - 1]

    def model_alone(self, antigen: Literal["s", "n"]) -> pm.Model:
        """
        Bayesian cox proportional hazards model. S or N titer is used as predictor of
        infection risk in subsequent month.

        Args:
            antigen: 's' (spike) or 'n' (nucleocapsid).
        """
        infected = convert_nan_to_zero(self.infected)
        exposure = convert_nan_to_zero(self.exposure)
        titer = {"s": self.s_titer, "n": self.n_titer}[antigen]
        coords = dict(intervals=self.intervals)
        with pm.Model(coords=coords) as model:
            lam0 = make_hierarchical_lam0(dims="intervals")
            a = pm.Normal("a", 0.0, 1.0)
            b = pm.Normal("b", 0.0, 1.0)
            lam = lam0 / (1.0 + np.exp((titer - a) @ -b))
            pm.Poisson("obs", exposure * lam, observed=infected)

        return model


def value_while_cumsum_below_threshold(
    values: Iterable[float], threshold: float = 1.0
) -> Generator[float, None, None]:
    """
    Generate values while their cumulative sum is below some threshold. The (maximum) sum
    out the output is always the threshold. An example is useful:

    Example:

        values = [0.09, 0.13, 0.15, 0.03, 0.24, 0.23, 0.19, 0.03, 0.25, 0.17]
        cumsum = [0.09, 0.22, 0.37, 0.4 , 0.64, 0.87, 1.06, 1.09, 1.34, 1.51]
        output = [0.09, 0.13, 0.15, 0.03, 0.24, 0.23, 0.13, 0.00, 0.00, 0.00]
                                                        |
                                Here there is only 0.13 'left' in the cumsum (from the
                                previous cumsum of 0.87) despite the value being 0.19.

    Args:
        values:
        threshold: Maximum sum of the output.
    """
    cumsum = 0.0
    for value in values:
        if value < 0.0:
            raise ValueError(f"values must be positive: {value}")
        elif cumsum > threshold:
            yield 0.0
        else:
            # if this value takes the cumulative sum over the threshold, yield whatever
            # is 'left' in the cumulative sum to get to the threshold, otherwise just
            # yield the value
            yield threshold - cumsum if cumsum + value > threshold else value

        cumsum += value


def make_nan_after_last_sample(
    arr: np.ndarray, last_gap: dict[int, int] | pd.Series
) -> np.ndarray:
    """
    Make entries in arr that are after the last sample taken for an individual nan.

    Args:
        arr: (n_inds, n_ints)
        last_sample: Mapping of individual i -> last gap.
    """
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(float)
    for ind in range(arr.shape[0]):
        if last_gap[ind] < 0:
            raise ValueError("values in last_gap must be positive")
        arr[ind, last_gap[ind] + 1 :] = np.nan
    return arr


def make_hierarchical_lam0(dims: list[str], **kwds) -> "pytensor.TensorVariable":
    """
    lam0 (lambda_0) is the baseline risk.

    Args:
        dims: Dimensions for lam0.
        **kwds passed to make_hierarchical normal.
    """
    _lam0_raw = make_hierarchical_normal(name="_lam0_raw", dims=dims, **kwds)
    return pm.Deterministic("lam0", pm.math.invlogit(_lam0_raw), dims=dims)


def make_hierarchical_normal(
    name: str,
    dims: str,
    hyper_mu: float = 0.0,
    hyper_sigma: float = 0.5,
    hyper_lam: float = 2.0,
    dims_mu: Optional[tuple[str, ...] | str] = None,
    dims_sigma: Optional[tuple[str, ...] | str] = None,
) -> "pytensor.tensor.TensorVariable":
    """
    Non-center parametrised hierarchical normal distribution. Equivalent to:

        mu = Normal(`name`_mu, mu=hyper_mu, sigma=hyper_sigma, dims=dims_mu)
        sigma = Exponential(`name`_sigma, lam=hyper_lam, dims=dims_sigma)
        Normal(name, mu=mu, sigma=sigma, dims=dims)

    Args:
        name: Variable name.
        dims: Dimensions of the model for the variable.
        hyper_{mu,sigma,lam}: Hyperpriors.
        dims_{mu,sigma}
    """
    mu = pm.Normal(f"{name}_mu", mu=hyper_mu, sigma=hyper_sigma, dims=dims_mu)
    sigma = pm.Exponential(f"{name}_sd", lam=hyper_lam, dims=dims_sigma)
    z = pm.Normal(f"_{name}_z", mu=0.0, sigma=1.0, dims=dims)
    return pm.Deterministic(name, z * sigma + mu, dims=dims)


def convert_nan_to_zero(arr: np.ndarray) -> np.ndarray:
    """Convert nan values to zero."""
    return np.where(np.isnan(arr), 0.0, arr)


def load_or_sample_model(
    path: str, model: pm.Model | Callable, *args, **kwargs
) -> az.InferenceData:
    """
    Try to load a pymc trace from a path. If the path doesn't exist then generate
    the trace by calling model(*args, **kwargs), save the inference data to path, and then
    return it.

    Args:
        path: Path to NetCDF file.
        model: A pymc Model or Callable that returns an arviz inference data object.
        *args, **kwargs: Passed to pymc.sample or the callable.
    """
    if not Path(path).suffix == ".nc":
        raise ValueError(f"file must end with .nc: {path}")

    if not (Path(path).parent.exists() and Path(path).parent.is_dir()):
        raise ValueError(f"directory does not exist: {path}")

    try:
        idata = az.from_netcdf(path)

    except FileNotFoundError:
        if isinstance(model, pm.Model):
            with model:
                idata = pm.sample(*args, **kwargs)
        else:
            idata = model(*args, **kwargs)

        idata.to_netcdf(path)

    finally:
        return idata
