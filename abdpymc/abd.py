#!/usr/bin/env python3

from typing import Union, Iterable, Optional
import math
import os
from abc import ABC

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor as pt
import pytensor.tensor as at

DARKORANGE = "#E14D2A"
BLUEGREY = "#3E6D9C"
GAP_IND = "gap", "ind"
IND_GAP = "ind", "gap"


class AntigenTiterData:
    def __init__(self, ag: str, df: pd.DataFrame):
        """
        Args:
            ag: One letter name of antigen e.g. S/N.
            df: DataFrame.
        """
        self.ag = ag
        self.df = df

        # Indexes to map inflection titers to the correct gaps and individuals
        self.idx_gap = df["elapsed_months"].values
        self.idx_ind = df["individual_i"].values

        # pymc distributions can't take np.int64 as shape parameters, hence int here
        self.n_gaps = int(max(self.idx_gap) + 1)
        self.n_inds = int(max(self.idx_ind) + 1)

    def __repr__(self) -> str:
        return f"AntigenTiterData(ag={self.ag}, df={self.df})"


class CombinedTiterData:
    def __init__(
        self, t0: pd.Period, df: pd.DataFrame, vacs: np.ndarray, pcrpos: np.ndarray
    ):
        """
        Manage OD, PCR+ and vaccination data for an antibody dyanamics analysis.

        Args:
            t0: The fist month in the analysis. All gap indexes are relative to this
                first month.
            n_inds: Only generate data for this many individuals.
            df: Don't make calls to the mfsera library to generate data, instead
                just load directly from this dataframe. See to_disk and from_disk.
            vacs: Similar to df, but for the vaccination data. See to_disk and from_disk.
            pcrpos: Similar to df, but for the PCR+ data. See to_disk and from_disk.

        Attributes:
            df: All the data. sample_i enumerates unique samples.
            df_s: Only the S1S2 data.
            df_n: Only the N data.

        In df_s and df_n the sample_{s,n}_i column enumerates unique samples that have
        been meausured for each antigen.
        """

        self.df = df

        # Map individual i to a record_id
        self.ind_i_to_record_id = dict(
            self.df[["individual_i", "record_id"]].drop_duplicates().values
        )

        self.s = AntigenTiterData(
            ag="s",
            df=(
                self.df.query("measurement == '10222020-S'")
                .copy()
                .assign(sample_s_i=lambda df: unique_ids(df["sample"]))
            ),
        )

        self.n = AntigenTiterData(
            ag="n",
            df=(
                self.df.query("measurement == '40588-V08B'")
                .copy()
                .assign(sample_n_i=lambda df: unique_ids(df["sample"]))
            ),
        )

        self.t0 = t0
        self.n_gaps = max(self.df["elapsed_months"]) + 1
        self.n_inds = max(self.df["individual_i"]) + 1

        self.ind_ids = list(
            tuple(v)
            for _, v in self.df[["individual_i", "record_id"]]
            .drop_duplicates()
            .iterrows()
        )

        # The following 2 arrays are (n_inds, n_gaps) that contain 1's if an event
        # (vaccination or PCR+) happened to an individual in a month, or 0 otherwise.
        self.vacs = vacs
        self.pcrpos = pcrpos

        self.plot_conf = PlotConfig(self)

        # gap index of the last sample for each indiviudal
        self.last_gap = (
            self.df.groupby("individual_i")
            .aggregate({"elapsed_months": "max"})
            .squeeze()
        )

        self.coords = dict(ind=np.arange(self.n_inds), gap=np.arange(self.n_gaps))

    def __repr__(self) -> str:
        return f"CombinedTiterData(t0={self.t0}, n_inds={self.n_inds})"

    @property
    def periods(self) -> pd.Series:
        """
        Month periods that correspond to each gap.
        """
        return pd.Series([self.t0 + i for i in range(self.n_gaps)])

    def date_to_gap(self, period: Union[str, pd.Period]) -> int:
        """
        Which gap did a particular date occur in?
        """
        return (pd.Period(period, freq="M") - self.t0).n

    def to_disk(self, directory: str) -> None:
        """
        Write a copy of self.df, self.vacs and self.pcrpos to a directory.

        Args:
            directory
        """
        if not os.path.exists(directory):
            os.mkdir(directory)

        fmt = "%1.0f"

        def path(x):
            return f"{directory}/{x}"

        self.df.to_csv(path("df.csv"))
        np.savetxt(path("vacs.txt"), self.vacs, fmt=fmt)
        np.savetxt(path("pcrpos.txt"), self.pcrpos, fmt=fmt)

        with open(path("t0.txt"), "w") as fobj:
            fobj.write(str(self.t0))

    @classmethod
    def from_disk(cls, directory: str) -> "CombinedTiterData":
        """
        Generate an instance of CombinedAllITitiersData by reading data from disk,
        rather than making calls to the mfsera package.

        Args:
            direcotry: Directory containint df.csv, vacs.txt, pcrpos.txt and t0.txt.
                See to_disk.
        """

        def path(x):
            return f"{directory}/{x}"

        df = pd.read_csv(path("df.csv"), index_col=0)
        vacs = np.loadtxt(path("vacs.txt"))
        pcrpos = np.loadtxt(path("pcrpos.txt"))

        if vacs.shape != pcrpos.shape:
            raise ValueError("vacs and pcrpos are different shapes")

        with open(path("t0.txt"), "r") as fobj:
            t0 = pd.Period(fobj.readline())

        return cls(t0=t0, df=df, vacs=vacs, pcrpos=pcrpos)

    def calculate_splits(self, delta: bool, omicron: bool) -> tuple[int]:
        """
        Given the time of the first gap, t0, return how many gaps exist between when
        delta and/or omicron started to circulate.

        Args:
            delta: Include the number of gaps to when delta started to circulate.
            omicron: Include the number of gaps to when omicron started to circulate.
        """
        splits = []

        if delta:
            splits.append((pd.Period("2021-07") - self.t0).n)

        if omicron:
            splits.append((pd.Period("2022-01") - self.t0).n)

        return tuple(splits)


def temp_response(exposure, n_inds, temp, rho):
    """
    Compute a temporary response.

    Args:
        exposure: (n_gaps, n_inds) matrix of infections, vaccinations or combination.
        n_inds: Number of individuals in the dataset.
        temp: Variable determining the size of the temporary response.
        rho: Gap to gap waning proportion.
    """
    response, _ = pt.scan(
        lambda e, prev, temp, rho: prev * rho + e * temp,
        sequences=[exposure],
        outputs_info=at.zeros(n_inds),
        non_sequences=[temp, rho],
    )
    return response


def perm_response(exposure, perm):
    """
    Compute a permanent reponse.

    The value is perm after the first exposure.

    Args:
        exposure: (n_gaps, n_inds) matrix of infections, vaccinations or combination.
        perm: The size of the permanent response.
    """
    return at.switch(at.cumsum(exposure, axis=0) > 0.0, perm, 0.0)


def model_popab_vacsep_nonwane_n(data: AntigenTiterData, i: at.TensorLike):
    """
    Model an N response.

    Features:
        - Only impacted by infection events.
        - Single temporary response parameter.
        - Single waning parameter.
        - All individuals wane.

    Args:
        data: Must have n_inds, idx_gap and idx_ind attributes.
        i: (n_gap, n_ind) array containing 0/1 indicating infection events.
    """
    assert data.ag == "n"

    def var(x):
        return f"ab_{data.ag}_{x}"

    # generate a value of perm for all gaps after the first infection in each individual
    perm = pm.Gamma(var("perm"), mu=2.0, sigma=0.5)
    perm_gap_ind = perm_response(exposure=i, perm=perm)

    # response that decays over time
    temp = pm.Gamma(var("temp"), mu=1.0, sigma=0.5)
    rho = pm.Beta(var("rho"), alpha=10.0, beta=1.0)

    temp_gap_ind = temp_response(exposure=i, n_inds=data.n_inds, temp=temp, rho=rho)

    init = pm.Normal(var("init"), -2, 1)
    mu = pm.Deterministic(var("mu"), perm_gap_ind + temp_gap_ind + init, dims=GAP_IND)

    return mu[data.idx_gap, data.idx_ind]


def model_popab_vacsep_nonwane_s(
    data: AntigenTiterData, i: at.TensorLike, v: at.TensorLike
):
    """
    Model an S response.

    Features:
        - Impacted by infection and vaccination events.
        - Separate temporary response parameters for vaccinations and infections.
        - Single waning parameter.
        - Individuals either wane (ab_n_waner=0) or do not (ab_n_waner=1)

    Args:
        data: Must have n_inds, idx_gap and idx_ind attributes.
        i: (n_gap, n_ind) array containing 0/1 indicating infection events.
        v: (n_gap, n_ind) array containing 0/1 indicating vaccinations.
    """
    assert data.ag == "s"

    def var(x):
        return f"ab_{data.ag}_{x}"

    # Permanent response after initial exposure (vaccination or infection)
    perm = pm.Gamma(var("perm"), mu=2.0, sigma=0.5)
    perm_gap_ind = perm_response(exposure=i + v, perm=perm)

    # response that decays over time for vaccinations and infections
    rho = pm.Beta(var("rho"), alpha=10.0, beta=1.0)
    p_waner = pm.Beta(var("p_waner"), alpha=1.0, beta=1.0)  # baseline p(waner)
    waner = pm.Bernoulli(var("waner"), p=p_waner, shape=data.n_inds)
    rho_per_ind = rho * waner + 1 - waner  # waner=1 -> rho=rho, waner=0 -> rho=1

    # Infection response
    tempinf = pm.Gamma(var("tempinf"), mu=1.0, sigma=0.5)
    tempinf_gap_ind = temp_response(
        exposure=i, n_inds=data.n_inds, temp=tempinf, rho=rho_per_ind
    )

    # Vaccination response
    tempvac = pm.Gamma(var("tempvac"), mu=1.0, sigma=0.5)
    tempvac_gap_ind = temp_response(
        exposure=v, n_inds=data.n_inds, temp=tempvac, rho=rho_per_ind
    )

    init = pm.Normal(var("init"), -2, 1)
    mu = pm.Deterministic(
        var("mu"), perm_gap_ind + tempinf_gap_ind + tempvac_gap_ind + init, dims=GAP_IND
    )

    return mu[data.idx_gap, data.idx_ind]


def make_time_chunks(
    pcrpos: at.TensorLike,
    splits: Optional[Union[tuple[int], tuple[int, int]]] = None,
):
    """
    Helper function to make a TimeChunks class based on the number of splits.
    """
    if splits is None or len(splits) == 0:
        return OneTimeChunk(pcrpos=pcrpos)

    elif len(splits) == 1:
        return TwoTimeChunks(split=splits[0], pcrpos=pcrpos)

    elif len(splits) == 2:
        return ThreeTimeChunks(splits=splits, pcrpos=pcrpos)

    else:
        raise NotImplementedError("only implemented 1-3 time chunks (0-2 splits)")


def dirichlet_multinomial_infections(
    n_gaps: int, n_inds: int, suffix: Union[str, int]
) -> at.TensorVariable:
    """
    Specify a (n_gaps x n_inds) array of infections.

    Implementation notes:
        Variables:
            - `p_any_i` is the probability that n=1 for across all individuals
            - `n` controls how many infections each individual gets in a given chunk of
              time gaps
            - `p` is the probability an individual gets infected in any given gap.
              Dirichlet across gaps means that probabilities sum to one across gaps for
              each individual.
            - `i` is a matrix of 0/1 indicating who got infected when

        To specify the dirichlet probabilities such that gaps sum to 1 have to specify
        (ind x gap) Dirichlet and Multinomial distributions. But everything else was
        already implemented on (gap x ind) arrays. Hence the need to transopose the `i`
        array here.

    Args:
        n_gaps: Number of time gaps.
        n_inds: Number of individuals.
        suffix: Suffix for the variable name and chunk dimension name.
    """
    p_any_i = pm.Beta(f"dmi_p_any_i_{suffix}", 1, 1)
    n = pm.Bernoulli(f"dmi_n_{suffix}", p_any_i, dims="ind")
    p = pm.Dirichlet(f"dmi_p_{suffix}", np.ones(n_gaps), shape=(n_inds, n_gaps))
    return pm.Multinomial(f"dmi_i_{suffix}", n=n, p=p, shape=(n_inds, n_gaps)).T


def model_dirmul(
    data: CombinedTiterData,
    splits: Union[tuple[int], tuple[int, int]],
    ignore_pcrpos: int = False,
) -> pm.model.Model:
    """
    Set up an antibody dynamics model that uses Dirichlet and Multinomial distributions
    to model infection probabilities.

    Args:
        data: Combined constant gap data.
        splits: A len 1 or len 2 tuple containing gap indexes for infection chunking.
        ignore_pcrpos: Should PCR+ data be ignored (e.g. for running a positive control
            test).
    """
    check_splits(splits=splits, data=data)

    # Vaccinations
    v = at.as_tensor(data.vacs.T)

    # PCR positives
    pcrpos = at.as_tensor(data.pcrpos.T)

    time_chunks = make_time_chunks(pcrpos=pcrpos, splits=splits)

    with pm.Model(coords=data.coords) as model:
        i = time_chunks.dirichlet_multinomial_infections

        # Ensure that in each chunk of time gaps that PCR+ data take precedence
        if not ignore_pcrpos:
            i = time_chunks.incorporate_pcrpos(i_raw=i)

        # Infer infections and ab parameters
        ititers_n = model_popab_vacsep_nonwane_n(data=data.n, i=i)
        ititers_s = model_popab_vacsep_nonwane_s(data=data.s, i=i, v=v)

        # Sigmoid likelihood
        model_sigmoids(data=data.n, a=ititers_n)
        model_sigmoids(data=data.s, a=ititers_s)

    return model


def model(
    data: CombinedTiterData,
    splits: Optional[Union[tuple[int], tuple[int, int]]] = None,
    ignore_pcrpos: int = False,
):
    """
    Set up an antibody dynamics model.

    Args:
        data: Combined constant gap data.
        splits: A len 1 or len 2 tuple containing gap indexes for infection chunking.
        ignore_pcrpos: Should PCR+ data be ignored (e.g. for running a positive control
            test).
    """
    check_splits(splits=splits, data=data)

    # Vaccinations
    v = at.as_tensor(data.vacs.T)

    # PCR positives
    pcrpos = (
        at.zeros_like(data.pcrpos.T) if ignore_pcrpos else at.as_tensor(data.pcrpos.T)
    )

    time_chunks = make_time_chunks(pcrpos=pcrpos, splits=splits)

    with pm.Model(coords=data.coords) as model:
        # Base infection probability. Loosely expect approx 1 infection per person.
        p = pm.Beta("p", alpha=1, beta=data.n_gaps - 1)

        # Exposure matrix
        i_raw = pm.Bernoulli("i_raw", p, dims=GAP_IND)

        # Constrain infections if need be w.r.t. how many infections are allowed per
        # time chunk, whether PCR+ infections take precedence over inferred infections
        # and prevent infections from occurring within 3 months of another infection.
        i = time_chunks.constrain_infections(i_raw)

        # Infer infections and ab parameters
        ititers_n = model_popab_vacsep_nonwane_n(data=data.n, i=i)
        ititers_s = model_popab_vacsep_nonwane_s(data=data.s, i=i, v=v)

        # Sigmoid likelihood
        model_sigmoids(data=data.n, a=ititers_n)
        model_sigmoids(data=data.s, a=ititers_s)

    return model


def model_sigmoids(data: AntigenTiterData, a: at.TensorLike) -> None:
    """
    Model sigmoid curves of each sample in data. Must be called from within a pymc model
    context.

    Args:
        data: AllITiters data. a: Inflection titers inferred from infection and antibody
        dynamics component of
            the model.
    """

    def var(x):
        return f"it_{data.ag}_{x}"

    pm.Normal(
        var("lik"),
        mu=logistic(
            x=data.df["log_dilution"].values,
            a=a,
            b=pm.Normal(var("b"), -1, 0.5),
            d=pm.Normal(var("d"), 2, 0.5),
        ),
        sigma=pm.Exponential(var("sigma"), 1),
        observed=data.df["od"].values,
    )


def unique_ids(values: Iterable) -> list[int]:
    """Give a unique integer starting from 0 to each value in values.

    Args:
        values
    """
    unique_values = set(values)
    mapping = dict(zip(unique_values, range(len(unique_values))))
    return [mapping[value] for value in values]


class PlotConfig:
    def __init__(self, data: CombinedTiterData):
        """
        Commonly used attributes in plotting indivdiual timeseries.
        """
        self.s = dict(it="ab_s_mu", c=DARKORANGE)
        self.n = dict(it="ab_n_mu", c=BLUEGREY)
        self.yticks = np.arange(0, 6, 2)
        self.yticklabels = [
            str(int(t)) for t in numeric_to_titer(self.yticks, start=40, fold=4)
        ]
        self.xticks, self.xticklabels = year_tick_labels(data.t0, data.n_gaps)


@np.vectorize
def numeric_to_titer(n, start, fold=4):
    """
    Convert a dilution to its concentration relative to the starting
    mixture based on its position in the dilution series.

    Notes:
        Imagine starting with a dilution of 1:40. If you conducted a
        four-fold dilution series the concentration would be:

        Concentration (parts) = 1:40, 1:160, 1:640, 1:2560, ...
        Position (n)          =    0,     1,     2,      3, ...

    Args:
        n (int): Position in dilution series (0-indexed).
        start (int): Initial dilution.
        fold (int): Fold difference of each subsequent dilution in series.
    """
    t = start * float(fold) ** n
    integer, fractional = math.modf(t)
    if fractional < 1e-6:
        return int(integer)
    else:
        return t


def months_from(t0: pd.Period, n: int) -> pd.Period:
    """Generate n month periods starting from t0"""
    curr = t0
    for _ in range(n):
        yield curr
        curr += pd.tseries.offsets.MonthEnd(1)


def year_tick_labels(t0: pd.Period, n_months: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate tick locations and labels to mark years starting from t0 for
    n_months.
    """
    ticks, labels = zip(
        *(
            (i, period.year)
            for i, period in enumerate(months_from(t0, n_months))
            if period.month == 1
        )
    )
    return np.array(ticks), np.array(labels)


def invlogistic(x, a, b, d):
    return a - np.log(d / x - 1) / b


def logistic(x, a, b, d):
    return d / (1 + np.exp(-b * (x - a)))


def mask_three_gaps(arr: at.TensorLike) -> at.TensorLike:
    """
    Prevent an infection from occuring within three months of another infection.

    Args:
        arr: (n_gaps, n_inds) tensor containing 1 if an individual was infected in a
            particular time gap.
    """
    _, n_inds = arr.shape.eval()
    taps = [-3, -2, -1]
    initial = at.zeros((len(taps), n_inds), dtype="int8")

    masked, _ = pt.scan(
        mask_future_infection,
        sequences=dict(input=at.cast(arr, "int8")),
        outputs_info=dict(taps=taps, initial=initial),
    )

    return masked


def mask_future_infection(i0, im3, im2, im1):
    """
    Prevent infection if infections have occured in any of the previous
    three months.

    All arguments are ints (0/1) that indicate whether an infection occurred in
    a certain timeslot.

    This function is passed to pytensor.scan. scan generates the next value based
    on the current value (i0) and 'taps' into other slots of the sequence. If

        -2 -1  0 +1 +2 +3
    i = [a, b, c, d, e, f]

    Args:
        i0: Current infections
        im3: Infection 3 months ago
        im2: Infection 2 months ago
        im1: Infection 1 month ago
    """
    return at.switch(im3 | im2 | im1, 0, i0)


def check_splits(
    splits: Union[tuple[int, ...], None], data: Optional[CombinedTiterData] = None
):
    """
    Check splits are valid.
    """
    if splits is not None:
        if any(split < 0 for split in splits):
            raise ValueError("split indexes must be positive")
        if sorted(splits) != list(splits):
            raise ValueError("splits must be in ascending order")
        if data is not None and splits and splits[-1] > data.n_gaps:
            raise ValueError(
                f"largest split must be less than n_gaps - 1, ({splits[-1]})"
            )
        if len(splits) != len(set(splits)):
            raise ValueError("splits not unique")
        if any(not isinstance(split, int) for split in splits):
            raise ValueError("splits must be ints")


class OneTimeChunk:
    def __init__(self, pcrpos: at.TensorLike) -> None:
        """
        A class that handles a single time chunk.

        In a single time chunk multiple infections are allowed, and PCR+ data do not
        take any special precendece over probabalistically inferred infections.

        Implementation note:
            This class does not need to inherit from the MultipleTimeChunks ABC as its
            constrain_infections method does not use mask_multiple_infections and
            incorporate_pcrpos.
        """
        self.pcrpos = pcrpos

    def constrain_infections(self, i_raw: at.TensorLike) -> at.TensorLike:
        # For a single time chunk, incorporating PCR positive data is as simple as
        # adding these two arrays.
        i_pcrpos = i_raw + self.pcrpos

        # It could be that 1s in i_raw and self.pcrpos collide
        # Here, ensure max value in i_pcrpos is 1
        i_pcrpos_max1 = at.where(i_pcrpos > 0.0, 1.0, 0.0)

        return pm.Deterministic("i", mask_three_gaps(i_pcrpos_max1))


class MultipleTimeChunks(ABC):
    """
    An abstract base class that defines how probabilistcally inferred infections and PCR+
    data are constrained.
    """

    def constrain_infections(self, i_raw: at.TensorLike) -> at.TensorLike:
        # Include PCR+ in infection matrix
        i_pcrpos = self.incorporate_pcrpos(i_raw)

        # Prevent multiple exposures in a single chunk
        i_single_inf_per_chunk = self.mask_multiple_infections(i_pcrpos)

        # Finally, prevent infections occuring within three months of each other,
        # regardless of which time chunk they occur in
        return pm.Deterministic("i", mask_three_gaps(i_single_inf_per_chunk))


class TwoTimeChunks(MultipleTimeChunks):
    def __init__(self, split: int, pcrpos: at.TensorLike) -> None:
        """
        A class for handling time broken into 2 different chunks. The two chunks are
        specified by passing an int.
        """
        if not isinstance(split, int):
            raise ValueError("1 split is required to specify 2 time chunks")
        else:
            self.split = split

        n_gaps, n_inds = pcrpos.shape.eval()
        self.n_gaps = int(n_gaps)
        self.n_inds = int(n_inds)

        self.pcrpos_0 = pcrpos[:split]
        self.pcrpos_1 = pcrpos[split:]

    def mask_multiple_infections(self, arr):
        return mask_multiple_infections_2_chunks(arr, split=self.split)

    def incorporate_pcrpos(self, i_raw):
        return at.concatenate(
            (
                incorporate_pcrpos(i_raw[: self.split], self.pcrpos_0),
                incorporate_pcrpos(i_raw[self.split :], self.pcrpos_1),
            )
        )

    @property
    def dirichlet_multinomial_infections(self) -> at.TensorVariable:
        """
        Separate Dirichlet / Multinomial infections for each chunk of time gaps.

        Args:
            n_gaps: The total number of gaps (for both time chunks).
            n_inds: Number of individuals.
        """
        a = dirichlet_multinomial_infections(
            n_inds=self.n_inds, n_gaps=self.split, suffix=0
        )
        b = dirichlet_multinomial_infections(
            n_inds=self.n_inds, n_gaps=self.n_gaps - self.split, suffix=1
        )
        return at.concatenate((a, b))


class ThreeTimeChunks(MultipleTimeChunks):
    def __init__(self, splits: tuple[int], pcrpos: at.TensorLike):
        """
        A class for handling time broken into 3 different chunks. The three chunks are
        specified by passing a tuple of 2 ints.
        """
        check_splits(splits)

        if len(splits) != 2:
            raise ValueError("2 splits are required to specify 3 time chunks")
        else:
            self.split0, self.split1 = splits

        self.pcrpos_0 = pcrpos[: self.split0]
        self.pcrpos_1 = pcrpos[self.split0 : self.split1]
        self.pcrpos_2 = pcrpos[self.split1 :]

    def mask_multiple_infections(self, arr):
        return mask_multiple_infections_3_chunks(
            arr, split0=self.split0, split1=self.split1
        )

    def incorporate_pcrpos(self, i_raw):
        return at.concatenate(
            (
                incorporate_pcrpos(i_raw[: self.split0], self.pcrpos_0),
                incorporate_pcrpos(i_raw[self.split0 : self.split1], self.pcrpos_1),
                incorporate_pcrpos(i_raw[self.split1 :], self.pcrpos_2),
            )
        )


def incorporate_pcrpos(i_raw: at.TensorLike, pcrpos: at.TensorLike):
    """
    Incorporate PCR positive data into raw infection.

    Raw infections are a matrix of 1s and 0s. Each row is a time gap, each column is an
    individual:

                i_raw = [
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ]

    PCR+ is encoded as a matrix of the same shape where a 1 indicates a PCR+ for a
    particular individual at a particular time:

                pcrpos = [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0],
                ]

    This function returns another matrix of the same shape. When an individual has a PCR+
    the column from the PCR+ is taken. When they don't have a PCR+ the column from i_raw
    is taken. This effectively removes any other infections for an individual when it is
    known they had a PCR+ in a particular time gap.

    Args:
        i_raw: (gap, ind)
        pcrpos: (gap, ind)
    """
    return at.where(pcrpos.any(axis=0), pcrpos, i_raw)


def mask_multiple_infections_3_chunks(arr, split0, split1):
    """
    A version of mask_multiple_infections_2_chunks, but with 2 splits / 3 chunks.

    Args:
        arr: The array containing infections.
        split0: The index where the second chunk begins.
        split1: The index where the third chunk begins.
    """
    return at.concatenate(
        (
            mask_multiple_infections(arr[:split0]),
            mask_multiple_infections(arr[split0:split1]),
            mask_multiple_infections(arr[split1:]),
        )
    )


def mask_multiple_infections(arr: at.TensorLike):
    """
    Prevent successive infections occuring.

    The following matrix encodes when indiviudals where infected. Each column is a
    different indiviudal, each row is a different time gap.

    [
        [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0]
    ]

    This function turns 1s -> 0s if they occur after another 1 in a column:

    [
        [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    Args:
        arr: (n gap, n individual) matrix encoding when an individual was infected.
    """
    return at.where(arr.cumsum(axis=0) > 1, 0, arr)


def mask_multiple_infections_2_chunks(arr: at.TensorLike, split: int):
    """
    A single infection is allowed within each chunk of the array.
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

    This should be the result:
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


    Args:
           arr: The array containing infections.
           split: The index where the second chunk begins. E.g. in the example above the
            index where chunk 2 starts is 4.
    """
    return at.concatenate(
        (mask_multiple_infections(arr[:split]), mask_multiple_infections(arr[split:]))
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("abd.py")
    parser.add_argument(
        "--model",
        help="Specifies the function that generates the pymc Model to use. Either "
        "'model' or 'model_dirmul'.",
        choices=("model", "model_dirmul"),
        required=True,
    )
    parser.add_argument(
        "--tune", help="Number of tuning steps.", type=int, required=True
    )
    parser.add_argument("--draws", help="Number of draws.", type=int, required=True)
    parser.add_argument("--cores", help="Number of cores", type=int)
    parser.add_argument(
        "--ititers_data",
        help="Path to directory for generating CombinedTiterData object.",
        default="cohort_data",
    )
    parser.add_argument(
        "--split_delta",
        help="Split time chunk between delta and pre-delta",
        action="store_true",
    )
    parser.add_argument(
        "--split_omicron",
        help="Split time chunk between omicron and delta",
        action="store_true",
    )
    parser.add_argument("--ignore_pcrpos", help="Ignore PCR+ data", action="store_true")
    parser.add_argument("--netcdf", help="Path of netCDF file to save.")
    args = parser.parse_args()

    data = CombinedTiterData.from_disk(args.ititers_data)

    splits = (
        None
        if (not args.split_delta) and (not args.split_omicron)
        else data.calculate_splits(delta=args.split_delta, omicron=args.split_omicron)
    )

    fun = {"model": model, "model_dirmul": model_dirmul}[args.model]

    with fun(data, splits=splits, ignore_pcrpos=args.ignore_pcrpos):
        idata = pm.sample(tune=args.tune, draws=args.draws, cores=args.cores)

    az.to_netcdf(idata, args.netcdf)
