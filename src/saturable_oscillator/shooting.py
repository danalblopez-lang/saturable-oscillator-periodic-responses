from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .model import Parameters, restoring_force, restoring_force_prime


@dataclass
class ShootingResult:
    initial_state: NDArray[np.float64]
    monodromy: NDArray[np.float64]
    residual_norm: float
    iterations: int
    multipliers: NDArray[np.complex128]
    liouville_defect: float


@dataclass
class OrbitProfile:
    time: NDArray[np.float64]
    displacement: NDArray[np.float64]
    velocity: NDArray[np.float64]


def augmented_rhs(
    t: float,
    y: NDArray[np.float64],
    parameters: Parameters,
    beta: float,
) -> NDArray[np.float64]:
    u, v = y[0], y[1]
    phi = y[2:].reshape(2, 2)
    jacobian = np.array(
        [
            [0.0, 1.0],
            [-restoring_force_prime(u, parameters.alpha), -parameters.delta],
        ],
        dtype=float,
    )

    derivative = np.empty_like(y)
    derivative[0] = v
    derivative[1] = (
        -parameters.delta * v
        - restoring_force(u, parameters.alpha)
        + beta * np.cos(parameters.omega * t)
    )
    derivative[2:] = (jacobian @ phi).ravel()
    return derivative


def integrate_augmented(
    initial_state: NDArray[np.float64],
    parameters: Parameters,
    beta: float,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    t_eval: NDArray[np.float64] | None = None,
):
    y0 = np.concatenate((np.asarray(initial_state, dtype=float), np.eye(2).ravel()))
    solution = solve_ivp(
        augmented_rhs,
        (0.0, parameters.period),
        y0,
        args=(parameters, beta),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
    )
    if not solution.success:
        raise RuntimeError(f"Time integration failed: {solution.message}")

    terminal_state = solution.y[:2, -1]
    monodromy = solution.y[2:, -1].reshape(2, 2)
    return terminal_state, monodromy, solution


def shooting_residual_and_jacobian(
    initial_state: NDArray[np.float64],
    parameters: Parameters,
    beta: float,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    terminal_state, monodromy, _ = integrate_augmented(
        initial_state,
        parameters,
        beta,
        rtol=rtol,
        atol=atol,
    )
    residual = terminal_state - initial_state
    jacobian = monodromy - np.eye(2)
    return residual, jacobian, monodromy


def solve_periodic_orbit(
    initial_guess: NDArray[np.float64],
    parameters: Parameters,
    beta: float,
    *,
    tolerance: float = 1.0e-11,
    maximum_iterations: int = 25,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> ShootingResult:
    state = np.asarray(initial_guess, dtype=float).copy()
    monodromy = np.eye(2)

    for iteration in range(1, maximum_iterations + 1):
        residual, jacobian, monodromy = shooting_residual_and_jacobian(
            state,
            parameters,
            beta,
            rtol=rtol,
            atol=atol,
        )
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        if residual_norm < tolerance:
            break

        try:
            correction = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "The shooting Jacobian is singular or ill-conditioned. "
                "Reduce the continuation step or use pseudo-arclength continuation."
            ) from exc

        step_length = 1.0
        accepted = False
        while step_length >= 2.0 ** -16:
            candidate = state + step_length * correction
            candidate_residual, _, candidate_monodromy = shooting_residual_and_jacobian(
                candidate,
                parameters,
                beta,
                rtol=rtol,
                atol=atol,
            )
            candidate_norm = float(np.linalg.norm(candidate_residual, ord=np.inf))
            if candidate_norm < (1.0 - 1.0e-4 * step_length) * residual_norm:
                state = candidate
                monodromy = candidate_monodromy
                accepted = True
                break
            step_length *= 0.5

        if not accepted:
            raise RuntimeError(
                f"Damped Newton iteration failed at beta={beta:.8g}; "
                f"last residual={residual_norm:.3e}."
            )
    else:
        raise RuntimeError(
            f"Newton iteration did not converge at beta={beta:.8g} "
            f"within {maximum_iterations} iterations."
        )

    final_residual, _, monodromy = shooting_residual_and_jacobian(
        state,
        parameters,
        beta,
        rtol=rtol,
        atol=atol,
    )
    final_norm = float(np.linalg.norm(final_residual, ord=np.inf))
    multipliers = np.linalg.eigvals(monodromy)
    liouville_target = np.exp(-parameters.delta * parameters.period)
    liouville_defect = float(abs(np.linalg.det(monodromy) - liouville_target))

    return ShootingResult(
        initial_state=state,
        monodromy=monodromy,
        residual_norm=final_norm,
        iterations=iteration,
        multipliers=np.asarray(multipliers, dtype=np.complex128),
        liouville_defect=liouville_defect,
    )


def continue_branch(
    initial_state: NDArray[np.float64],
    beta_values: Iterable[float],
    parameters: Parameters,
    *,
    tolerance: float = 1.0e-11,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> list[ShootingResult]:
    state = np.asarray(initial_state, dtype=float).copy()
    results: list[ShootingResult] = []
    for beta in beta_values:
        result = solve_periodic_orbit(
            state,
            parameters,
            float(beta),
            tolerance=tolerance,
            rtol=rtol,
            atol=atol,
        )
        results.append(result)
        state = result.initial_state.copy()
    return results


def state_rhs(
    t: float,
    state: NDArray[np.float64],
    parameters: Parameters,
    beta: float,
) -> NDArray[np.float64]:
    u, v = state
    return np.array(
        [
            v,
            -parameters.delta * v
            - restoring_force(u, parameters.alpha)
            + beta * np.cos(parameters.omega * t),
        ],
        dtype=float,
    )


def sample_orbit(
    initial_state: NDArray[np.float64],
    parameters: Parameters,
    beta: float,
    *,
    number_of_points: int = 2001,
    include_endpoint: bool = True,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
) -> OrbitProfile:
    time = np.linspace(
        0.0,
        parameters.period,
        number_of_points,
        endpoint=include_endpoint,
    )
    solution = solve_ivp(
        state_rhs,
        (0.0, parameters.period),
        np.asarray(initial_state, dtype=float),
        args=(parameters, beta),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=time,
    )
    if not solution.success:
        raise RuntimeError(f"Orbit integration failed: {solution.message}")
    return OrbitProfile(
        time=time,
        displacement=solution.y[0],
        velocity=solution.y[1],
    )
