# abdpymc

Antibody dynamics using PyMC.

GitHub repo:
[github.com/davipatti/abdpymc](https://github.com/davipatti/abdpymc).

## Installation

To install, clone and `cd` to this repo. Make a new virtual environment and  `pip install .`.

To run tests do `pip install .[dev]` and then call `pytest`.

## Usage

The main entry points are `abdpymc-infer` to run inference on data and `abdpymc-plot-timelines` to
plot individual timelines.

To see this repo in action, see [our paper](https://www.cell.com/iscience/fulltext/S2589-0042(25)01647-5):

> Pattinson, D. J. et al. **Emergence of SARS-CoV-2 Omicron lineages shifted
> antibody-mediated protection curves**. *iScience* 113386 (2025)
> doi:10.1016/j.isci.2025.113386.

All the data and code from that paper are in [this
repository](https://doi.org/10.17632/r7675pg8hf.1).