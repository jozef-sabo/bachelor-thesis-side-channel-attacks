# Side-Channel Attacks on Blinded Scalar Multiplications — Cleanroom Implementation

This repository contains the source code accompanying the bachelor's thesis *Correcting errors in the side-channel attacks* (Masaryk University, Faculty of Informatics, 2026). It provides a cleanroom Python reimplementation of the divide-and-conquer side-channel attack of Roche, Imbert, and Lomné (CARDIS 2019), together with four proposed optimizations and a parameterizable trace generator.

## What it does

Given multiple noisy observations of a blinded scalar `d_l = r_l · E + d`, where `E` is the order of an elliptic curve of structured form `E = 2^K − E_0` (e.g. secp256k1), the algorithms recover the secret scalar `d` bit by bit using a beam search over candidate prefixes of the blinding factors `r_l`.

## Repository layout

```
algorithm/
  article_algo.py          # cleanroom baseline (Roche–Imbert–Lomné attack)
  elliptic_curves.py       # curve definitions (secp256k1, P-256, Curve25519, ...)
  numbers_generator.py     # CLI tool for generating synthetic noisy traces
improved_algorithm/
  calibrated_error_rate_algo.py                            # ε_e calibration
  boundary_retention_algo.py                               # inclusive / capped boundary retention
  multi_bit_search_window_effective_guess_window_algo.py   # multi-bit lookahead
traces/                    # pre-generated trace datasets (.pkl)
```

## Requirements

- Python 3.10+
- NumPy (only for trace generation in `numbers_generator.py --original` mode)

```bash
pip install numpy
```

No other third-party dependencies — the attack itself runs on the standard library.

## Generating traces

```bash
cd algorithm
python numbers_generator.py secp256k1 10000 traces.pkl \
    --error-rate 15 \
    --multiplier-bits 64
```

Arguments:

| Argument | Description |
|---|---|
| `curve` | One of `secp256k1`, `secp256r1`, `P256`, `W25519`, `Curve25519`, `Edwards25519` |
| `count` | Number of traces to generate |
| `out_file` | Output pickle file |
| `-e, --error-rate` | Bit error rate as integer percentage (default `15`) |
| `-m, --multiplier-bits` | Blinding factor length `R` (default `64`) |
| `-o, --original` | Use the trace-generation methodology from the original paper (normal-distribution thresholding + restricted `r` interval) instead of uniform Bernoulli noise |

## Running the baseline attack

```python
from algorithm.article_algo import run_dataset

bits_recovered = run_dataset(
    file_path="traces.pkl",
    N=10000,   # number of traces to use
    L=32,      # candidate-list size
    t=16,      # evaluation window size
)
print(bits_recovered)
```

The function returns the number of correctly recovered least-significant bits of `d` before the first incorrect bit, under a supervised early-termination criterion.

## Running the optimized variants

Each optimized variant exposes a `run_dataset` with the same baseline parameters plus its own knobs.

**Calibrated error rate** — overrides the assumed bit-error rate inside `evaluate_probability`:

```python
from improved_algorithm.calibrated_error_rate_algo import run_dataset
run_dataset("traces.pkl", N=10000, L=32, t=16, error_rate_set=22)  # ε_e = 0.22
```

**Inclusive boundary retention** — retains candidates tied at the truncation boundary:

```python
from improved_algorithm.boundary_retention_algo import run_dataset
run_dataset(
    "traces.pkl", N=10000, L=32, t=16,
    rs_with_same_prob_at_border="keep",   # "keep" | "remove" | "not_touch"
    three_fourths=False,                  # True enables the 7L/4 cap
)
```

**Multi-bit search window with optional asymmetric commitment**:

```python
from improved_algorithm.multi_bit_search_window_effective_guess_window_algo import run_dataset
run_dataset(
    "traces.pkl", N=10000, L=32, t=16,
    d_add_window=4,        # w_d — search window width
    d_widening_window=4,   # w_de — commitment window (≤ w_d)
)
```

## Trace file format

Each `.pkl` file is a Python dictionary with the following fields:

| Key | Description |
|---|---|
| `curve_size` | Curve bit length `K` |
| `E` | Curve order |
| `d` | True secret scalar (for supervised evaluation) |
| `multiplier_size` | Blinding factor length `R` |
| `error_rate` | Bit error rate (integer percentage) |
| `blinded_with_errors` | List of `N` noisy blinded scalars `d̃_l` |
| `multipliers` | List of true blinding factors `r_l` |
| `error_vectors` | List of true error vectors `ε_l` |

## License

Released under the MIT License. See `LICENSE` for details.
