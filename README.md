# Saturable oscillator periodic responses

Reproducible numerical continuation, asymptotic expansions, and Floquet diagnostics for periodic responses of the forced saturable oscillator

\[
u''+\delta u'+u-\alpha\frac{u}{\sqrt{1+u^2}}=\beta\cos(\omega t).
\]

## Contents

The repository implements:

- continuation of the three branches from the exact constant states at `beta = 0`;
- Newton shooting with the shooting Jacobian obtained from the variational equation;
- Floquet multipliers and the Liouville determinant defect;
- first-, second-, and central cubic asymptotic approximations;
- an independent centered finite-difference periodic solver with sparse Newton correction;
- CSV tables and publication-ready PDF/PNG figures used in the manuscript.

## Installation

Python 3.11 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the package and its dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Reproduce the manuscript outputs

```bash
python scripts/reproduce_paper.py
```

The default parameter set is

```text
alpha = 2.0
delta = 0.35
omega = 1.0
beta in [0, 0.40]
continuation step = 0.01
representative beta = 0.20
```

Generated files are written to

```text
outputs/figures/
outputs/tables/
```

The file `outputs/tables/generated_tables.tex` is generated from the same numerical data as the CSV files.

## Repository structure

```text
src/saturable_oscillator/   numerical library
scripts/                    reproduction script
outputs/figures/            PDF and high-resolution PNG figures
outputs/tables/             CSV and LaTeX tables
paper/                      numerical section used in the manuscript
tests/                      automated tests
```

## Numerical scope

The continuation routine uses the previously converged shooting solution as the next predictor and applies a damped Newton corrector. It is suitable while `beta` remains a regular parameter along the branch. Near a fold, the code stops rather than silently crossing an ill-conditioned point; pseudo-arclength continuation would then be required.

The finite-difference calculations, residuals, symmetry defects, Liouville defects, and mesh-refinement indicators are reproducibility diagnostics. They are not interval-certified existence proofs.

## Tests

```bash
python -m pip install pytest
pytest -q
```

## Citation

Use the citation information in `CITATION.cff`. After the GitHub release is archived in Zenodo, cite the version-specific Zenodo DOI associated with the manuscript.

## License

The source code is distributed under the MIT License. See `LICENSE`.
