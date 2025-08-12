from importlib.metadata import version, PackageNotFoundError

from .abd import (
    BLUEGREY,
    TiterData,
    compute_chunked_cum_p,
    DARKORANGE,
    incorporate_pcrpos,
    invlogistic,
    logistic,
    mask_future_infection,
    mask_multiple_infections_2_chunks,
    mask_multiple_infections_3_chunks,
    mask_multiple_infections,
    mask_three_gaps,
    model,
    scalar_variables,
    TwoTimeChunks,
)
from . import abd
from . import plotting
from . import simulation
from . import timelines
from . import survival
from .survival import load_or_sample_model, SurvivalAnalysis


try:
    __version__ = version("abdpymc")
except PackageNotFoundError:
    pass

__all__ = [
    "abd",
    "BLUEGREY",
    "TiterData",
    "compute_chunked_cum_p",
    "DARKORANGE",
    "incorporate_pcrpos",
    "invlogistic",
    "load_or_sample_model",
    "logistic",
    "mask_future_infection",
    "mask_multiple_infections_2_chunks",
    "mask_multiple_infections_3_chunks",
    "mask_multiple_infections",
    "mask_three_gaps",
    "model",
    "plotting",
    "scalar_variables",
    "simulation",
    "survival",
    "SurvivalAnalysis",
    "temp_response",
    "timelines",
    "TwoTimeChunks",
]
