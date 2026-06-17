#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from saturable_oscillator.asymptotics import (  # noqa: E402
    central_cubic_approximation,
    first_order_approximation,
    second_order_approximation,
)
from saturable_oscillator.finite_difference import (  # noqa: E402
    nested_grid_discrepancy,
    solve_periodic_finite_difference,
)
from saturable_oscillator.model import Parameters, zero_forcing_states  # noqa: E402
from saturable_oscillator.shooting import (  # noqa: E402
    continue_branch,
    sample_orbit,
)


BRANCH_NAMES = ("minus", "zero", "plus")
BRANCH_LABELS = {"minus": r"$u_-$", "zero": r"$u_0$", "plus": r"$u_+$"}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_complex_latex(value: complex) -> str:
    if abs(value.imag) < 5.0e-10:
        return f"{value.real:.9g}"
    sign = "+" if value.imag >= 0 else "-"
    return f"{value.real:.8f}{sign}{abs(value.imag):.8f}\\,\\mathrm{{i}}"


def continue_all_branches(
    parameters: Parameters,
    beta_values: np.ndarray,
    number_of_profile_points: int,
):
    states = zero_forcing_states(parameters.alpha)
    continuation = {}
    summaries: list[dict] = []

    for branch_name, xi in zip(BRANCH_NAMES, states, strict=True):
        results = continue_branch(
            np.array([xi, 0.0]),
            beta_values,
            parameters,
        )
        continuation[branch_name] = results

        for beta, result in zip(beta_values, results, strict=True):
            profile = sample_orbit(
                result.initial_state,
                parameters,
                float(beta),
                number_of_points=number_of_profile_points,
            )
            multipliers = result.multipliers
            summaries.append(
                {
                    "branch": branch_name,
                    "beta": float(beta),
                    "u0": result.initial_state[0],
                    "v0": result.initial_state[1],
                    "u_min": float(profile.displacement.min()),
                    "u_max": float(profile.displacement.max()),
                    "mu1_real": multipliers[0].real,
                    "mu1_imag": multipliers[0].imag,
                    "mu1_abs": abs(multipliers[0]),
                    "mu2_real": multipliers[1].real,
                    "mu2_imag": multipliers[1].imag,
                    "mu2_abs": abs(multipliers[1]),
                    "shooting_residual": result.residual_norm,
                    "liouville_defect": result.liouville_defect,
                    "newton_iterations": result.iterations,
                }
            )

    return continuation, summaries


def representative_diagnostics(
    parameters: Parameters,
    beta_values: np.ndarray,
    continuation,
    representative_beta: float,
):
    index = int(np.argmin(np.abs(beta_values - representative_beta)))
    if not np.isclose(beta_values[index], representative_beta, atol=1.0e-14):
        raise ValueError("representative_beta must belong to the continuation grid")

    profiles = {}
    rows: list[dict] = []
    for branch_name in BRANCH_NAMES:
        result = continuation[branch_name][index]
        profile = sample_orbit(
            result.initial_state,
            parameters,
            representative_beta,
            number_of_points=4001,
        )
        profiles[branch_name] = profile
        multipliers = result.multipliers
        stable = bool(np.max(np.abs(multipliers)) < 1.0)
        classification = "stable" if stable else "unstable"
        rows.append(
            {
                "branch": branch_name,
                "u0": result.initial_state[0],
                "v0": result.initial_state[1],
                "u_min": float(profile.displacement.min()),
                "u_max": float(profile.displacement.max()),
                "mu1_real": multipliers[0].real,
                "mu1_imag": multipliers[0].imag,
                "mu1_abs": abs(multipliers[0]),
                "mu2_real": multipliers[1].real,
                "mu2_imag": multipliers[1].imag,
                "mu2_abs": abs(multipliers[1]),
                "det_monodromy": float(np.linalg.det(result.monodromy)),
                "shooting_residual": result.residual_norm,
                "liouville_defect": result.liouville_defect,
                "classification": classification,
            }
        )

    return profiles, rows


def symmetry_diagnostics(parameters: Parameters, beta: float, representative_rows: list[dict]):
    profiles = {}
    number_of_points = 4096
    for row in representative_rows:
        initial_state = np.array([row["u0"], row["v0"]])
        profile = sample_orbit(
            initial_state,
            parameters,
            beta,
            number_of_points=number_of_points,
            include_endpoint=False,
        )
        profiles[row["branch"]] = profile.displacement

    half_shift = number_of_points // 2
    exterior_defect = float(
        np.max(
            np.abs(
                profiles["minus"]
                + np.roll(profiles["plus"], -half_shift)
            )
        )
    )
    central_defect = float(
        np.max(
            np.abs(
                profiles["zero"]
                + np.roll(profiles["zero"], -half_shift)
            )
        )
    )
    return [
        {"identity": "u_minus(t)+u_plus(t+T/2)", "defect": exterior_defect},
        {"identity": "u_zero(t)+u_zero(t+T/2)", "defect": central_defect},
    ]


def asymptotic_diagnostics(
    parameters: Parameters,
    beta_values: np.ndarray,
    continuation,
    test_betas: tuple[float, ...],
):
    states = dict(zip(BRANCH_NAMES, zero_forcing_states(parameters.alpha), strict=True))
    rows: list[dict] = []
    for beta in test_betas:
        index = int(np.argmin(np.abs(beta_values - beta)))
        if not np.isclose(beta_values[index], beta, atol=1.0e-14):
            raise ValueError(f"beta={beta} must belong to the continuation grid")

        for branch_name in BRANCH_NAMES:
            result = continuation[branch_name][index]
            profile = sample_orbit(
                result.initial_state,
                parameters,
                beta,
                number_of_points=4001,
            )
            xi = states[branch_name]
            approximation_1 = first_order_approximation(
                profile.time,
                beta,
                xi,
                parameters,
            )
            error_1 = float(np.max(np.abs(profile.displacement - approximation_1)))

            if branch_name == "zero":
                approximation_high = central_cubic_approximation(
                    profile.time,
                    beta,
                    parameters,
                )
                error_high = float(
                    np.max(np.abs(profile.displacement - approximation_high))
                )
                rows.append(
                    {
                        "branch": branch_name,
                        "beta": beta,
                        "first_order_error": error_1,
                        "higher_order_error": error_high,
                        "first_normalized": error_1 / beta**3,
                        "higher_normalized": error_high / beta**5,
                        "higher_order": 3,
                    }
                )
            else:
                approximation_high = second_order_approximation(
                    profile.time,
                    beta,
                    xi,
                    parameters,
                )
                error_high = float(
                    np.max(np.abs(profile.displacement - approximation_high))
                )
                rows.append(
                    {
                        "branch": branch_name,
                        "beta": beta,
                        "first_order_error": error_1,
                        "higher_order_error": error_high,
                        "first_normalized": error_1 / beta**2,
                        "higher_normalized": error_high / beta**3,
                        "higher_order": 2,
                    }
                )
    return rows


def finite_difference_diagnostics(
    parameters: Parameters,
    beta: float,
    representative_rows: list[dict],
    mesh_sizes: tuple[int, ...],
):
    initial_states = {
        row["branch"]: np.array([row["u0"], row["v0"]])
        for row in representative_rows
    }
    solutions: dict[tuple[int, str], np.ndarray] = {}
    intervals: list[dict] = []

    for number_of_nodes in mesh_sizes:
        time = np.arange(number_of_nodes) * parameters.period / number_of_nodes
        for branch_name in BRANCH_NAMES:
            profile = sample_orbit(
                initial_states[branch_name],
                parameters,
                beta,
                number_of_points=number_of_nodes,
                include_endpoint=False,
            )
            result = solve_periodic_finite_difference(
                profile.displacement,
                parameters,
                beta,
            )
            solutions[(number_of_nodes, branch_name)] = result.displacement
            intervals.append(
                {
                    "N": number_of_nodes,
                    "branch": branch_name,
                    "u_min": float(result.displacement.min()),
                    "u_max": float(result.displacement.max()),
                    "residual": result.residual_norm,
                    "newton_iterations": result.iterations,
                }
            )

    refinement: list[dict] = []
    for coarse_size, fine_size in zip(mesh_sizes[:-1], mesh_sizes[1:], strict=True):
        row = {"transition": f"{coarse_size}->{fine_size}"}
        for branch_name in BRANCH_NAMES:
            row[branch_name] = nested_grid_discrepancy(
                solutions[(coarse_size, branch_name)],
                solutions[(fine_size, branch_name)],
            )
        refinement.append(row)

    return intervals, refinement


def plot_continuation(output_dir: Path, summaries: list[dict]) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for branch_name in BRANCH_NAMES:
        rows = [row for row in summaries if row["branch"] == branch_name]
        beta = np.array([row["beta"] for row in rows])
        u_min = np.array([row["u_min"] for row in rows])
        u_max = np.array([row["u_max"] for row in rows])
        axis.plot(beta, u_min, label=f"{BRANCH_LABELS[branch_name]}: min")
        axis.plot(beta, u_max, linestyle="--", label=f"{BRANCH_LABELS[branch_name]}: max")
    axis.set_xlabel(r"Forcing amplitude $\beta$")
    axis.set_ylabel(r"Extrema over one period")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "branch_continuation.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "branch_continuation.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_representative_orbits(output_dir: Path, profiles, beta: float) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for branch_name in BRANCH_NAMES:
        profile = profiles[branch_name]
        axis.plot(
            profile.time,
            profile.displacement,
            label=BRANCH_LABELS[branch_name],
        )
    axis.set_xlabel(r"Time $t$")
    axis.set_ylabel(r"Displacement $u(t)$")
    axis.set_title(rf"Periodic responses at $\beta={beta:.2f}$")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "representative_orbits.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "representative_orbits.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_floquet_moduli(output_dir: Path, summaries: list[dict]) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for branch_name in BRANCH_NAMES:
        rows = [row for row in summaries if row["branch"] == branch_name]
        beta = np.array([row["beta"] for row in rows])
        spectral_radius = np.array(
            [max(row["mu1_abs"], row["mu2_abs"]) for row in rows]
        )
        axis.plot(beta, spectral_radius, label=BRANCH_LABELS[branch_name])
    axis.axhline(1.0, linestyle="--", label=r"$|\mu|=1$")
    axis.set_yscale("log")
    axis.set_xlabel(r"Forcing amplitude $\beta$")
    axis.set_ylabel(r"Spectral radius of the monodromy matrix")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "floquet_moduli.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "floquet_moduli.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def write_generated_tables(
    path: Path,
    representative_rows: list[dict],
    symmetry_rows: list[dict],
    asymptotic_rows: list[dict],
    intervals: list[dict],
    refinement: list[dict],
) -> None:
    representative = {row["branch"]: row for row in representative_rows}
    interval_lookup = {(row["N"], row["branch"]): row for row in intervals}

    with path.open("w", encoding="utf-8") as handle:
        handle.write("% Generated by scripts/reproduce_paper.py. Do not edit manually.\n\n")
        handle.write("\\begin{table}[ht]\n\\centering\n")
        handle.write("\\caption{Shooting and Floquet diagnostics at $\\beta=0.20$.}\n")
        handle.write("\\label{tab:representative-floquet}\n\\small\n")
        handle.write("\\begin{tabular}{lrrrrrr}\n\\hline\n")
        handle.write("Branch & $u(0)$ & $u'(0)$ & $\\min u$ & $\\max u$ & Multipliers & Class \\\\\n\\hline\n")
        for branch_name in BRANCH_NAMES:
            row = representative[branch_name]
            mu1 = complex(row["mu1_real"], row["mu1_imag"])
            mu2 = complex(row["mu2_real"], row["mu2_imag"])
            handle.write(
                f"{BRANCH_LABELS[branch_name]} & {row['u0']:.6f} & {row['v0']:.6f} & "
                f"{row['u_min']:.6f} & {row['u_max']:.6f} & "
                f"${format_complex_latex(mu1)},\\ {format_complex_latex(mu2)}$ & "
                f"{row['classification']} \\\\\n"
            )
        handle.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")

        handle.write("\\begin{table}[ht]\n\\centering\n")
        handle.write("\\caption{Residual and structural diagnostics at $\\beta=0.20$.}\n")
        handle.write("\\label{tab:representative-diagnostics}\n\\small\n")
        handle.write("\\begin{tabular}{lrrr}\n\\hline\n")
        handle.write("Branch & Shooting residual & Liouville defect & $\\det M$ \\\\\n\\hline\n")
        for branch_name in BRANCH_NAMES:
            row = representative[branch_name]
            handle.write(
                f"{BRANCH_LABELS[branch_name]} & {row['shooting_residual']:.3e} & "
                f"{row['liouville_defect']:.3e} & {row['det_monodromy']:.9f} \\\\\n"
            )
        handle.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")

        handle.write("\\begin{table}[ht]\n\\centering\n")
        handle.write("\\caption{Numerical defects in the exact half-period symmetry identities.}\n")
        handle.write("\\label{tab:symmetry-defects}\n\\small\n")
        handle.write("\\begin{tabular}{lr}\n\\hline\nIdentity & Uniform defect \\\\\n\\hline\n")
        for row in symmetry_rows:
            handle.write(f"\\texttt{{{row['identity']}}} & {row['defect']:.3e} \\\\\n")
        handle.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")

        handle.write("\\begin{table}[ht]\n\\centering\n")
        handle.write("\\caption{Uniform errors of the local asymptotic approximations.}\n")
        handle.write("\\label{tab:asymptotic-errors}\n\\small\n")
        handle.write("\\begin{tabular}{lrrrr}\n\\hline\n")
        handle.write("Branch & $\\beta$ & First-order error & Higher-order error & Higher-order normalized \\\\\n\\hline\n")
        for row in asymptotic_rows:
            handle.write(
                f"{BRANCH_LABELS[row['branch']]} & {row['beta']:.3f} & "
                f"{row['first_order_error']:.3e} & {row['higher_order_error']:.3e} & "
                f"{row['higher_normalized']:.3e} \\\\\n"
            )
        handle.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")

        handle.write("\\begin{table}[ht]\n\\centering\n")
        handle.write("\\caption{Finite-difference amplitude intervals under mesh refinement.}\n")
        handle.write("\\label{tab:mesh-intervals-new}\n\\small\n")
        handle.write("\\begin{tabular}{cccc}\n\\hline\n")
        handle.write("$N$ & $u_N^-$ & $u_N^0$ & $u_N^+$ \\\\\n\\hline\n")
        for number_of_nodes in sorted({row["N"] for row in intervals}):
            values = []
            for branch_name in BRANCH_NAMES:
                row = interval_lookup[(number_of_nodes, branch_name)]
                values.append(f"[{row['u_min']:.6f},{row['u_max']:.6f}]")
            handle.write(
                f"{number_of_nodes} & ${values[0]}$ & ${values[1]}$ & ${values[2]}$ \\\\\n"
            )
        handle.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")

        handle.write("\\begin{table}[ht]\n\\centering\n")
        handle.write("\\caption{Nested-grid maximum discrepancies.}\n")
        handle.write("\\label{tab:mesh-refinement-new}\n\\small\n")
        handle.write("\\begin{tabular}{lrrr}\n\\hline\n")
        handle.write("Transition & $u_-$ & $u_0$ & $u_+$ \\\\\n\\hline\n")
        for row in refinement:
            handle.write(
                f"${row['transition'].replace('->', '\\to')}$ & {row['minus']:.4e} & "
                f"{row['zero']:.4e} & {row['plus']:.4e} \\\\\n"
            )
        handle.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the numerical results for the forced saturable oscillator."
    )
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=0.35)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--beta-max", type=float, default=0.40)
    parser.add_argument("--beta-step", type=float, default=0.01)
    parser.add_argument("--representative-beta", type=float, default=0.20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs",
    )
    args = parser.parse_args()

    parameters = Parameters(alpha=args.alpha, delta=args.delta, omega=args.omega)
    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    number_of_steps = int(round(args.beta_max / args.beta_step))
    beta_values = np.linspace(0.0, args.beta_max, number_of_steps + 1)

    continuation, summaries = continue_all_branches(
        parameters,
        beta_values,
        number_of_profile_points=1001,
    )
    write_csv(
        table_dir / "branch_continuation.csv",
        list(summaries[0].keys()),
        summaries,
    )

    profiles, representative_rows = representative_diagnostics(
        parameters,
        beta_values,
        continuation,
        args.representative_beta,
    )
    write_csv(
        table_dir / "representative_floquet.csv",
        list(representative_rows[0].keys()),
        representative_rows,
    )

    symmetry_rows = symmetry_diagnostics(
        parameters,
        args.representative_beta,
        representative_rows,
    )
    write_csv(
        table_dir / "symmetry_diagnostics.csv",
        list(symmetry_rows[0].keys()),
        symmetry_rows,
    )

    asymptotic_rows = asymptotic_diagnostics(
        parameters,
        beta_values,
        continuation,
        test_betas=(0.020, 0.050, 0.100),
    )
    write_csv(
        table_dir / "asymptotic_errors.csv",
        list(asymptotic_rows[0].keys()),
        asymptotic_rows,
    )

    intervals, refinement = finite_difference_diagnostics(
        parameters,
        args.representative_beta,
        representative_rows,
        mesh_sizes=(128, 256, 512),
    )
    write_csv(
        table_dir / "mesh_intervals.csv",
        list(intervals[0].keys()),
        intervals,
    )
    write_csv(
        table_dir / "mesh_refinement.csv",
        list(refinement[0].keys()),
        refinement,
    )

    plot_continuation(figure_dir, summaries)
    plot_representative_orbits(figure_dir, profiles, args.representative_beta)
    plot_floquet_moduli(figure_dir, summaries)

    write_generated_tables(
        table_dir / "generated_tables.tex",
        representative_rows,
        symmetry_rows,
        asymptotic_rows,
        intervals,
        refinement,
    )

    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
