# abdpymc

[![abdpymc tests](https://github.com/davipatti/abdpymc/actions/workflows/run-tests.yml/badge.svg)](https://github.com/davipatti/abdpymc/actions/workflows/run-tests.yml)

Antibody dynamics using PyMC.

[Github](https://github.com/davipatti/abdpymc) repo.

## Installation

To install, clone and `cd` to this repo. Make a new virtual environment and  `pip install .`.

To run tests do `pip install .[dev]` and then call `pytest`.

## Usage

The main entry points are `abdpymc-infer` to run inference on data and `abdpymc-plot-timelines` to
plot individual timelines.

To see this repo in action, see [this repository](https://doi.org/10.17632/r7675pg8hf.1).