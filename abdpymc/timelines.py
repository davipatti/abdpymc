#!/usr/bin/env python3

from typing import Iterable, Union, Optional, Callable, Literal
import time
import functools
import logging
import string
import warnings

import xarray as xr
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import abdpymc as abd

Number = Union[float, int]


def grouper(iterable, n):
    """
    Inspired by:

    >>> def grouper(iterable: Iterable, n: int, fillvalue: Any = None):
    >>>    '''Group items in iterable in size n batches'''
    >>>    args = [iter(iterable)] * n
    >>>    return itertools.zip_longest(*args, fillvalue=fillvalue)

    The above grouper pads batches with fillvalue, e.g.:

    >>> list(grouper("David Pattinson", 6))
    >>> [('D', 'a', 'v', 'i', 'd', ' '),
    >>> ('P', 'a', 't', 't', 'i', 'n'),
    >>> ('s', 'o', 'n', None, None, None)]

    Wheras this version does not pad the final group with a fillvalue.
    """
    values = tuple(iterable)
    start = 0
    while start < len(values):
        yield values[start : start + n]
        start += n


def _hi_lo_scaler_scalar(p: Number, lo: Number, hi: Number):
    """Scale a proportion p between lo and hi."""
    return p * (hi - lo) + lo


def hi_lo_scaler(p: Union[Number, Iterable[Number]], lo: Number, hi: Number):
    """
    Plotting helper function.

    Scale a proportion p between the values lo and hi.
    """
    try:
        return [_hi_lo_scaler_scalar(_p, lo, hi) for _p in p]
    except TypeError:
        return _hi_lo_scaler_scalar(p, lo, hi)


def load_ititers(path: str, data: abd.CombinedTiterData) -> pd.DataFrame:
    """Load DataFrame containing inflection titers"""
    df = pd.read_csv(path, index_col=0)
    df["elapsed_months"] = [data.date_to_gap(date) for date in df["collection_date"]]
    return df


def gap_jitter(i: int, j: int) -> np.array:
    """
    Plotting helper funciton.

    Add some horizontal jitter to antibody responses over time.

    Args:
        i: Number of rows.
        j: Number of columns.

    Returns:
        (i, j) array. Each column has the same amount of jitter added. Values of each
            row grow by 1 each row. E.g.:

            >>> gap_jitter(4, 6)
            array([[0.46, 0.81, 0.25, 0.48, 0.28, 0.31],
                   [1.46, 1.81, 1.25, 1.48, 1.28, 1.31],
                   [2.46, 2.81, 2.25, 2.48, 2.28, 2.31],
                   [3.46, 3.81, 3.25, 3.48, 3.28, 3.31]])
    """
    return (
        np.random.uniform(-0.5, 0.5, j).reshape(-1, 1)
        + np.tile(np.arange(0.5, i + 0.5), j).reshape(j, i)
    ).T


def plot_events(
    events_binary: np.ndarray,
    ymin: Union[float, int],
    ymax: Union[float, int],
    **kwds,
):
    """
    Plot PCR+ or vaccination events as bars.

    Args:
        events_binary: (n_gaps,) Array containing 1s in the gaps where the events
            occurred.
        ymin: Bottom of the bar.
        ymax: Top of the bar.
        **kwds: Passed to plt.vlines.
    """
    events, *_ = np.nonzero(events_binary)
    plt.bar(
        events,
        height=ymax - ymin,
        width=1,
        bottom=ymin,
        align="edge",
        zorder=0,
        **kwds,
    )


def plot_variant_background(
    start: pd.Period,
    t0: pd.Period,
    width: int,
    ax=None,
    label=None,
    fontsize=6,
    ymin=0,
    ymax=1,
    bleed=0.75,
    **kwds,
):
    """
    Shaded rectangle showing when a particular variant circulated.

    Args:
        start: The first month this variant became predominant.
        t0: The first month on the ax.
        width: Width of the rectangle
        ax:
        label: Text label for the patch.
        fontsize:
        ymin:
        ymax:
        bleed: Amount the patch extends underneath the xaxis.
        **kwds: Passed to mpl.patches.Rectangle.
    """
    ax = plt.gca() if ax is None else ax
    x = (start - t0).n

    ax.add_artist(
        mpl.patches.Rectangle(
            xy=(x, ymin - bleed),
            width=width,
            height=ymax - ymin + bleed,
            clip_on=False,
            zorder=-1,
            **kwds,
        )
    )

    if label:
        ax.text(
            x + width / 2,
            ymax + 0.02,
            label,
            ha="center",
            va="bottom",
            zorder=20,
            fontsize=fontsize,
        )


plot_omicron_background = functools.partial(
    plot_variant_background,
    start=pd.Period("2022-01"),
    width=11,
    facecolor="#e0deed",
    label="Omicron",
    bleed=0,
)


plot_delta_background = functools.partial(
    plot_variant_background,
    start=pd.Period("2021-07"),
    width=6,
    facecolor="#fee7cd",
    label="Delta",
    bleed=0,
)


plot_predelta_background = functools.partial(
    plot_variant_background,
    start=pd.Period("2020-05"),
    width=14,
    facecolor="#daf1ed",
    label="pre-Delta",
    bleed=0,
)

plot_preomicron_background = functools.partial(
    plot_variant_background,
    start=pd.Period("2020-05"),
    width=20,
    facecolor="#daf1ed",
    label="pre-Omicron",
    bleed=0,
)


def plot_individual(
    ind_i,
    data: abd.CombinedTiterData,
    post: xr.Dataset,
    df_ititers: pd.DataFrame,
    ymin: Union[float, int] = -4,
    ymax: Union[float, int] = 8,
    post_skip: Optional[int] = None,
    splits: Optional[tuple[int]] = None,
    post_mean: Optional[xr.DataArray] = None,
    show_s_post: bool = True,
    show_n_post: bool = True,
    show_s_data: bool = True,
    show_n_data: bool = True,
    show_inf_prob: bool = True,
    show_cuminf_prob: bool = True,
    label_fontsize: float = 6,
    show_only_1_40_without_ititers: bool = True,
):
    """
    Plot posterior distribution of antibody responses for individual i in post.

    Args:
        ind_i: The index of the individual to plot in post.
        data: Data used to run ab dynamics model.
        post: Posterior distribution. chain and draw should be combined, e.g.:
            idata.posterior.stack({"sample": ("chain", "draw")})
        df_ititers: DataFrame containing inflection titers to plot.
        ymin: Minimum y value.
        ymax: Maximum y value.
        post_skip: Skip this many lines when plotting raw samples from the posterior
            distribution. If None, then calculate a value such that 50 posterior samples
            are plotted.
        splits: Plot cummulative infection probabilities within the time chunks delimited
            by splits.
        post_mean: Precomupted posterior mean. Can save time to precomupte over multiple
            individuals if plot_individual is being called alot.
        show_s_post: Show posterior infection probabilities and antibody responses for S.
        show_n_post: Show posterior infection probabilities and antibody responses for N.
        show_s_data: Show inflection titer estimates for S.
        show_n_data: Show inflection titer estimates for N.
        show_inf_prob: Stair plot showing monthly infection probability.
        show_cuminf_prob: Stair plot showing cumulative monthly infection probability.
        label_fontsize: Fontsize for labelling variant periods, individual record number
            and index.
        show_only_1_40_without_ititers: Only show 1:40 data where there isn't inflection
            titer data too.
    """
    ax = plt.gca()

    scale = functools.partial(hi_lo_scaler, lo=ymin, hi=ymax)

    record_id = data.ind_i_to_record_id[ind_i]
    df_ind_i = data.df.query("individual_i == @ind_i").set_index("elapsed_months")

    last_gap = data.last_gap[ind_i]

    post_skip = len(post.sample) // 250 if post_skip is None else post_skip

    if show_s_post or show_n_post:
        ags = {(True, False): "s", (False, True): "n", (True, True): "sn"}[
            (show_s_post, show_n_post)
        ]
        for ag in ags:
            dic = getattr(data.plot_conf, ag)
            color = dic["c"]
            it_name = dic["it"]  # 'ab_s_mu' or 'ab_n_mu'

            ititer = getattr(post, it_name)
            response = ititer[:last_gap, ind_i]

            # Plot mean response
            response_mean = (
                getattr(post_mean, it_name)[:last_gap, ind_i]
                if post_mean is not None
                else response.mean(dim="sample")
            )
            gap_mids = np.arange(0.5, last_gap + 0.5)
            ax.plot(gap_mids, response_mean, c="white", lw=2, zorder=12)
            ax.plot(gap_mids, response_mean, c=color, lw=2, ls="--", zorder=14)

            # Posterior samples
            y = response[:, ::post_skip]
            x = gap_jitter(*y.shape)
            ax.plot(x, y, c=color, lw=0.5, alpha=0.1)

    # Stairs showing monthly infection probability
    if show_inf_prob or show_cuminf_prob:
        inf_p = (
            post.sel(gap=slice(None, last_gap), ind=ind_i).mean(dim="sample").i
            if post_mean is None
            else post_mean.sel(gap=slice(None, last_gap), ind=ind_i).i
        )

    if show_inf_prob:
        ax.stairs(
            scale(inf_p),
            edges=np.arange(last_gap + 2),
            color="black",
            baseline=None,
            clip_on=False,
            zorder=10,
        )

    if show_cuminf_prob:
        cum_p = abd.compute_chunked_cum_p(inf_p, splits=splits)
        plt.stairs(
            scale(cum_p),
            edges=np.arange(0, len(cum_p) + 1),
            baseline=None,
            color="grey",
            clip_on=False,
            zorder=9,
            lw=0.5,
        )

    # PCR+ and vaccinations
    kwds = dict(ymin=ymin, ymax=ymax)
    plot_events(data.pcrpos[ind_i], color="#f2aaa8", **kwds)
    plot_events(data.vacs[ind_i], color="#afd0aa", **kwds)

    # Background patches
    kwds = dict(t0=data.t0, ymin=ymin, ymax=ymax, alpha=0.25, fontsize=label_fontsize)
    plot_preomicron_background(**kwds)
    plot_omicron_background(**kwds)

    # ELISA titers
    if show_s_data or show_n_data:
        ind_i_samples = df_ind_i["sample"].unique()
        ind_i_ititer_samples = df_ititers.index.intersection(ind_i_samples)

        kwds = dict(
            ax=ax,
            df=df_ititers.loc[ind_i_ititer_samples],
            distinguish_single_vs_multiple=False,
            hdi=True,
        )

        if show_s_data:
            plot_titers(antigen="10222020-S", color=data.plot_conf.s["c"], **kwds)

        if show_n_data:
            plot_titers(antigen="40588-V08B", color=data.plot_conf.n["c"], **kwds)

    # Individual index and record ID
    ax.text(
        0,
        1.01,
        f"i:{ind_i} r:{record_id}",
        transform=ax.transAxes,
        fontsize=label_fontsize,
    )

    # Finish ax
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, data.n_gaps)

    finish_timeline_ax(ax, data)

    return ax


def plot_titers(
    df: pd.DataFrame,
    antigen: Literal["10222020-S", "40588-V08B"],
    ax: mpl.axes.Axes,
    color: str,
    distinguish_single_vs_multiple: bool = False,
    hdi: bool = True,
):
    """
    Plot titers inferred independently of the antibody dynamics framework.

    Args:
        df: DataFrame containing the titers.
        antigen: The catalog number of the antigen to plot.
        ax: Matplotlib ax.
        color: Color for the data.
        distinguish_single_multiple: Use different markers for titers that were inferred
            using a single dilution vs. those inferred using multiple dilutions.
        hdi: Show the HDI.
    """
    kwds = dict(c=color, lw=0.5, zorder=15)

    df = df.query(f"antigen == '{antigen}'")

    if distinguish_single_vs_multiple:
        df_single = df.query("dilutions == '40'")
        # if not df_single.empty:
        ax.scatter(
            df_single["elapsed_months"] + 0.5,
            df_single["titer"],
            marker="x",
            s=5,
            **kwds,
        )

        df_multiple = df.query("dilutions != '40'")
        # if not df_multiple.empty:
        ax.scatter(
            df_multiple["elapsed_months"] + 0.5,
            df_multiple["titer"],
            marker="o",
            s=20,
            ec="white",
            **kwds,
        )
    else:
        ax.scatter(
            df["elapsed_months"] + 0.5,
            df["titer"],
            marker="o",
            s=20,
            ec="white",
            **kwds,
        )

    if hdi:
        ax.errorbar(
            x=df["elapsed_months"] + 0.5,
            y=df["titer"],
            yerr=np.stack([df["titer"] - df["hdi_lo"], df["hdi_hi"] - df["titer"]]),
            fmt="none",
            elinewidth=0.5,
            c=color,
            alpha=0.5,
            capsize=1.0,
            capthick=0.5,
        )


def finish_timeline_ax(
    ax: mpl.axes.Axes, data: abd.CombinedTiterData, yaxis=True
) -> None:
    """
    Adjust ticks and labels for a plot which has cohort time on the x-axis.

    Args:
        ax: Matplotlib ax.
        data: Cohort data object.
        yaxis: Whether to setup the y-axis to display titers.
    """
    ax.grid(zorder=0)
    ax.xaxis.set_minor_locator(mpl.ticker.IndexLocator(base=3, offset=2))
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=12, offset=2))
    ax.tick_params(which="minor", length=4)
    ax.tick_params(which="major", length=8, axis="x")
    if yaxis:
        ax.set_yticks(data.plot_conf.yticks, data.plot_conf.yticklabels)
    ax.set_xticks(data.plot_conf.xticks, data.plot_conf.xticklabels)
    ax.minorticks_off()


def plot_multiple_individuals(
    nrows: int,
    ncols: int,
    ax_width: int = 4,
    individuals: Optional[tuple[int, ...]] = None,
    subplots_kwds: Optional[dict] = None,
    lettered_labels: Optional[bool] = False,
    **kwds,
):
    """
    Plot a grid of antibody timeseries for multiple individuals.

    Args:
        nrows: Number of rows.
        ncols: Number of cols
        ax_width: Width of an individual plot.
        individuals: Indexes of individuals to plot.
        subplots_kwds: Dict of keyword arguments to pass to plt.subplots.
        lettered_labels: Add letter labels to the top left of each ax.
        **kwds passed to plot_individual.
    """
    width = ax_width * ncols
    height = ax_width / 1.618 * nrows

    fig, _ = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        figsize=(width, height),
        **subplots_kwds,
    )

    individuals = range(nrows * ncols) if individuals is None else individuals

    for i, ind_i in enumerate(individuals):

        ax = fig.axes[i]

        plt.sca(ax)
        plot_individual(ind_i=ind_i, **kwds)

        if lettered_labels:
            ax.text(
                0.02,
                0.98,
                string.ascii_uppercase[i],
                weight="bold",
                transform=ax.transAxes,
                va="top",
                fontsize=16,
            )


def plot_multiple_individuals_batches(
    individuals: Iterable[int],
    fname: str,
    nrows: int = 10,
    ncols: int = 25,
    **kwds,
):
    """
    Plot multiple individual timelines in batches of up to 250.

    Args:
        individuals: Indexes of individuals to plot.
        fname: Should contain two '{}' to for the first and last individual in the group.
        nrows: Number of rows of axes.
        ncols: Number of columns of axes.
        **kwds: Passed to plot_multiple_individuals.
    """
    batch_size = nrows * ncols

    for batch in grouper(individuals, batch_size):
        plot_multiple_individuals(nrows=nrows, ncols=ncols, individuals=batch, **kwds)
        plt.savefig(fname.format(batch[0], batch[-1]), bbox_inches="tight", dpi=300)
        plt.close()


def log_time(func: Callable) -> Callable:
    """Timing decorator"""

    def wrapped(*args, **kwargs):
        t0 = time.time()
        logging.log(logging.INFO, msg=f"Calling {func.__name__}", end=" ")
        result = func(*args, **kwargs)
        t1 = time.time()
        logging.log(logging.INFO, msg=f"total = {t1-t0:2.1f} s")
        return result

    return wrapped


@log_time
def stack_posterior(idata: az.InferenceData) -> xr.Dataset:
    """Combine chain and draw dimensions into single sample dimension."""
    return idata.posterior.stack({"sample": ("chain", "draw")})


@log_time
def compute_posterior_mean(post: xr.Dataset) -> xr.Dataset:
    """Compute mean accross samples."""
    return post.mean(dim="sample")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        "abd-timelines",
        description="Default behaviour is to plot all individuals in batches of 250. "
        "Use --individuals to plot only specific individuals or --pcrpos_only to plot "
        "only individuals that have a PCR+ test result.",
    )
    parser.add_argument("--idata", help="Path of .nc file", required=True)
    parser.add_argument(
        "--png_pref",
        help="Prefix of PNG filenames to save. Files will have this suffix appended: "
        "'-{}-{}.png'. The index of the first and last individuals occupy the curly "
        "braces.",
    )
    parser.add_argument(
        "--cohort_data",
        help="Path to directory for generating CombinedAllITitersData object "
        "(default=cohort_data).",
        default="cohort_data",
        required=True,
    )
    parser.add_argument(
        "--no_cumm_p",
        action="store_true",
        help="Dont plot cummulative infection probabilities. --split_{delta,omicron} "
        "have no effect if this is passed.",
    )
    parser.add_argument(
        "--split_delta",
        action="store_true",
        help="Plot cummulative infection probabilities in separate chunks. Use this "
        "flag to split delta from pre-delta.",
    )
    parser.add_argument(
        "--split_omicron",
        action="store_true",
        help="Like --split_delta but for splitting omicron from delta.",
    )
    parser.add_argument(
        "--individuals",
        help="Plot timelines for these specific individuals.",
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--nrows",
        help="Number of rows when passing specific individuals (default=19).",
        type=int,
        default=19,
    )
    parser.add_argument(
        "--ncols",
        help="Number of columns when passing specific individuals (default=16).",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--fname",
        help="Name(s) of file(s) to save when passing specific individuals. Can "
        "specify multiple names to plot, say PNG and PDF.",
        nargs="+",
    )
    parser.add_argument(
        "--pcrpos_only",
        help="Only plot individuals who have a PCR+ result. Ignored if --individuals "
        "is passed.",
        action="store_true",
    )
    parser.add_argument(
        "--lettered_labels",
        action="store_true",
        help="If using --individuals, add alphabetical labels to each ax.",
    )
    args = parser.parse_args()

    idata = az.from_netcdf(args.idata)

    data = abd.CombinedTiterData.from_disk(args.cohort_data)

    splits = (
        data.calculate_splits(delta=args.split_delta, omicron=args.split_omicron)
        if not args.no_cumm_p
        else None
    )

    post = stack_posterior(idata)

    post_mean = compute_posterior_mean(post)

    df_ititers = load_ititers(path=f"{args.cohort_data}/ititers.csv", data=data)

    kwds = dict(
        data=data,
        post=post,
        post_mean=post_mean,
        splits=splits,
        df_ititers=df_ititers,
        subplots_kwds=dict(gridspec_kw=dict(wspace=0.05, hspace=0.15)),
    )

    if args.individuals:
        if len(args.individuals) != args.nrows * args.ncols:
            warnings.warn(
                f"""
                Mismatch between number of individuals ({len(args.individuals)}) and
                number of rows ({args.nrows}) and columns ({args.ncols}). Use --nrows and
                --ncols.
                """
            )

        plot_multiple_individuals(
            individuals=args.individuals,
            nrows=args.nrows,
            ncols=args.ncols,
            lettered_labels=args.lettered_labels,
            **kwds,
        )
        for path in args.fname:
            plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    elif args.pcrpos_only:
        plot_multiple_individuals_batches(
            individuals=np.flatnonzero(data.pcrpos.any(axis=1)),
            fname=f"{args.png_pref}-{{}}-{{}}.png",
            **kwds,
        )

    else:
        plot_multiple_individuals_batches(
            individuals=range(data.n_inds),
            fname=f"{args.png_pref}-{{}}-{{}}.png",
            nrows=args.nrows,
            ncols=args.ncols,
            **kwds,
        )


if __name__ == "__main__":
    main()
