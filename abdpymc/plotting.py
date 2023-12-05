from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def overhanging_slice(values: Iterable, center: int, pad: float, fill=None) -> tuple:
    values = tuple(values)

    if center < 0:
        raise ValueError("center must be positive")

    if center > len(values):
        raise ValueError("center must be less than or equal to length of values")

    if pad < 0:
        raise ValueError("pad < 0")

    output = [fill] * (1 + 2 * pad)

    for i_output, i_values in enumerate(range(center - pad, center + pad + 1)):
        if 0 <= i_values < len(values):
            output[i_output] = values[i_values]

    return tuple(output)


def relative_to_known_ticks(
    window: int, ax: Optional[mpl.axes.Axes] = None, label: str = "PCR+"
) -> None:
    """
    Set xticks that show time of the known infection with positive and negative month
    offsets. This assumes that the known infection was in the center.

    Args:
        window: Width of the window.
        ax: Matplotlib ax.
        label: The label to put in the center.
    """
    if window % 2 != 1:
        raise ValueError("window should be odd")

    ax = plt.gca() if ax is None else ax
    labels = [""] * window

    labels[window // 2] = label

    for i, tick in enumerate(range(window // 2 + 1, window), start=1):
        labels[tick] = f"+{i}"

    for i, tick in enumerate(reversed(range(window // 2)), start=1):
        labels[tick] = f"−{i}"

    ax.set_xticks(np.arange(window), labels)


def infp_known_centered(p_infection: np.ndarray, known: np.ndarray, pad: float):
    """
    Construct an array where p_infection are aligned relative to all known infections.

    Args:
        p_infection: (n_gap, n_ind) array containing infection probabilities.
        known: (n_gap, n_ind) array of known infections. Must be same shape as
            p_infection.
        pad: Value to fill arrays if pad extends outside of p_infection.
    """
    if p_infection.shape != known.shape:
        raise ValueError(
            f"p_infection shape {p_infection.shape} != known shape {known.shape}"
        )

    return np.array(
        [
            overhanging_slice(p_infection[:, ind], center=gap, pad=pad, fill=np.nan)
            for gap, ind in np.argwhere(known)
        ]
    )


def plot_p_infection_relative_to_known_infection(
    p_infection: np.ndarray,
    known_infection: np.ndarray,
    pad: int,
    points: bool = False,
    lines: bool = False,
    mean_line: bool = False,
    ax: Optional[mpl.axes.Axes] = None,
    xlabel: str = "Months since PCR+",
    xtick_center_label: str = "PCR+",
) -> None:
    """
    Plot infection probabilities relative to known infections. All known infections are
    centered in the plot and infection probabilities relative the known infections are
    shown `pad` months before and after.

    Args:
        p_infection: (n_gap, n_ind) array containing infection probabilities.
        known_infection: (n_gap, n_ind) array of known infections. Must be same shape as
            p_infection.
        pad: Value to fill arrays if pad extends outside of p_infection.
        points: Add points on lines.
        lines: Plot all lines.
        mean_line: Show the mean of all infection probabilities.
        ax: Matplotlib axes.
        xlabel: X-label.
        xtick_center_label: Label for the central xtick.
    """
    ax = plt.gca() if ax is None else ax

    infp_overlaid = infp_known_centered(
        p_infection=p_infection, known=known_infection, pad=pad
    )

    n_known, window = infp_overlaid.shape

    # each known infection gets its own level of jitter
    jitter = np.tile(np.random.uniform(-0.25, 0.25, n_known), (window, 1)).T

    x = np.arange(window)

    x_tile = np.tile(x, (n_known, 1))

    kwds = dict(clip_on=False)

    if points:
        ax.scatter(x_tile + jitter, infp_overlaid, alpha=0.1, lw=0, zorder=12, **kwds)

    if lines:
        ax.plot(
            x_tile.T + jitter.T,
            infp_overlaid.T,
            alpha=0.05,
            zorder=10,
            c="grey",
            **kwds,
        )

    if mean_line:
        ax.plot(
            x,
            np.nanmean(infp_overlaid, axis=0),
            linewidth=2,
            marker="o",
            markersize=5,
            c="black",
            zorder=15,
            **kwds,
        )

    ax.axvline(window // 2, c="black", ls="--")

    relative_to_known_ticks(window, ax=ax, label=xtick_center_label)

    ax.set(ylim=(0, 1), ylabel="P(Infection)", xlabel=xlabel)
