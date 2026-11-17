# PMT-GLASSO

Python implementation of pairwise measure-transformed graphical lasso for
*Structure-Preserving Adaptive Measure Transforms for Robust High-Dimensional Graphical Models*.

## Setup

Run these commands from the repository root using Python 3.11 or newer.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Fit a graph

```python
import numpy as np
from pmtgl.covariance import pmt_cov
from pmtgl.glasso import glasso
from pmtgl.psd import eigclip

X = np.random.default_rng(0).standard_normal((100, 8))
S, info = pmt_cov(X)
d = np.sqrt(np.diag(S))
scale = np.outer(d, d)
R = eigclip(S / scale, rel=1e-3)
Theta = glasso(R, lam=0.2) / scale
```

`X` has observations in rows and variables in columns. `info["a"]` contains
the fitted transform strengths. Set `centre="mt"` for MT recentering.
PMT estimates correlations pairwise, so the example clips the spectrum
before fitting with block coordinate descent.

## Reproduce the experiments

The `pmtgl/` package contains covariance estimators, graphical lasso solvers,
PSD corrections, tuning rules, simulation tools and population formulas.
The `experiments/` scripts fit models and generate study summaries.
The `tests/` directory checks population identities, solver behavior and tuning.

To run a small synthetic tuning experiment, use:

```sh
mkdir -p results
python experiments/run_tuning2.py results/smoke.csv 1 40 8 er2 clean PMT
```

For the full suite, prepare the two gene-expression inputs using
[data/README.md](data/README.md), then run `bash reproduce.sh`.
The full suite can take many hours. Raw CSVs go in `results/` and generated
tables and figures go in `paper/tables/` and `paper/figures/`.
Several scripts use the existing CSVs to resume interrupted runs.
Use a fresh checkout or move the existing outputs aside for a clean run.

The [data notes](data/README.md) identify the sources and filenames for each
real-data experiment. Simulations and numerical tests use synthetic data.
