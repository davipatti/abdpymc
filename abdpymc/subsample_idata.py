#!/usr/bin/env python3

import argparse

import numpy as np
import arviz as az


def main():
    parser = argparse.ArgumentParser("abdpymc-subsample-idata")
    parser.add_argument(
        "--idata", required=True, help="NetCDF file of InferenceData object."
    )
    parser.add_argument("--output", required=True, help="Name of NetCDF path to write.")
    parser.add_argument(
        "--n", required=True, type=int, help="Number of samples per chain to subsample."
    )
    args = parser.parse_args()

    idata = az.from_netcdf(args.idata)

    draws = idata.sample_stats.dims["draw"]

    if args.n > draws:
        raise ValueError(
            f"asking for {args.n} samples but there are only {draws} in total"
        )

    samples = np.round(np.linspace(start=0, stop=draws - 1, num=args.n)).astype(int)

    az.to_netcdf(idata.sel(draw=samples), args.output)


if __name__ == "__main__":
    main()
