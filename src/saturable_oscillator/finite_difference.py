from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve

from .model import Parameters, restoring_force, restoring_force_prime


@dataclass
class FiniteDifferenceResult:
    displacement: NDArray[np.float64]
    residual_norm: float
    iterations: int


def periodic_difference_matrices(number_of_nodes: int, period: float):
    step = period / number_of_nodes
    first = lil_matrix((number_of_nodes, number_of_nodes), dtype=float)
    second = lil_matrix((number_of_nodes, number_of_nodes), dtype=float)

    for index in range(number_of_nodes):
        previous_index = (index - 1) % number_of_nodes
        next_index = (index + 1) % number_of_nodes
        first[index, next_index] = 1.0 / (2.0 * step)
        first[index, previous_index] = -1.0 / (2.0 * step)
        second[index, next_index] = 1.0 / step**2
        second[index, index] = -2.0 / step**2
        second[index, previous_index] = 1.0 / step**2

    return first.tocsr(), second.tocsr()


def discrete_residual(
    displacement: NDArray[np.float64],
    parameters: Parameters,
    beta: float,
    first_matrix,
    second_matrix,
) -> NDArray[np.float64]:
    number_of_nodes = displacement.size
    time = np.arange(number_of_nodes) * parameters.period / number_of_nodes
    forcing = beta * np.cos(parameters.omega * time)
    return (
        second_matrix @ displacement
        + parameters.delta * (first_matrix @ displacement)
        + restoring_force(displacement, parameters.alpha)
        - forcing
    )


def solve_periodic_finite_difference(
    initial_guess: NDArray[np.float64],
    parameters: Parameters,
    beta: float,
    *,
    tolerance: float = 1.0e-10,
    maximum_iterations: int = 25,
) -> FiniteDifferenceResult:
    displacement = np.asarray(initial_guess, dtype=float).copy()
    number_of_nodes = displacement.size
    first_matrix, second_matrix = periodic_difference_matrices(
        number_of_nodes,
        parameters.period,
    )

    for iteration in range(1, maximum_iterations + 1):
        residual = discrete_residual(
            displacement,
            parameters,
            beta,
            first_matrix,
            second_matrix,
        )
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        if residual_norm < tolerance:
            return FiniteDifferenceResult(displacement, residual_norm, iteration)

        jacobian = (
            second_matrix
            + parameters.delta * first_matrix
            + diags(restoring_force_prime(displacement, parameters.alpha))
        )
        correction = spsolve(jacobian, -residual)

        step_length = 1.0
        accepted = False
        while step_length >= 2.0 ** -14:
            candidate = displacement + step_length * correction
            candidate_residual = discrete_residual(
                candidate,
                parameters,
                beta,
                first_matrix,
                second_matrix,
            )
            candidate_norm = float(np.linalg.norm(candidate_residual, ord=np.inf))
            if candidate_norm < (1.0 - 1.0e-4 * step_length) * residual_norm:
                displacement = candidate
                accepted = True
                break
            step_length *= 0.5

        if not accepted:
            raise RuntimeError(
                f"Finite-difference Newton iteration failed for N={number_of_nodes}."
            )

    residual = discrete_residual(
        displacement,
        parameters,
        beta,
        first_matrix,
        second_matrix,
    )
    raise RuntimeError(
        f"Finite-difference Newton iteration did not converge for N={number_of_nodes}; "
        f"residual={np.linalg.norm(residual, ord=np.inf):.3e}."
    )


def nested_grid_discrepancy(
    coarse: NDArray[np.float64],
    fine: NDArray[np.float64],
) -> float:
    if fine.size != 2 * coarse.size:
        raise ValueError("The fine grid must contain twice as many nodes as the coarse grid.")
    return float(np.max(np.abs(coarse - fine[::2])))
