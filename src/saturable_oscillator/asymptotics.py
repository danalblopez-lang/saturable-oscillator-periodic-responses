from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import Parameters, restoring_force_prime, restoring_force_second


def complex_linear_response(xi: float, parameters: Parameters) -> complex:
    kappa = restoring_force_prime(xi, parameters.alpha)
    denominator = kappa - parameters.omega**2 + 1j * parameters.delta * parameters.omega
    return 1.0 / denominator


def first_order_correction(
    time: ArrayLike,
    xi: float,
    parameters: Parameters,
) -> NDArray[np.float64]:
    time_array = np.asarray(time, dtype=float)
    z = complex_linear_response(xi, parameters)
    return np.real(z * np.exp(1j * parameters.omega * time_array))


def second_order_correction(
    time: ArrayLike,
    xi: float,
    parameters: Parameters,
) -> NDArray[np.float64]:
    time_array = np.asarray(time, dtype=float)
    kappa = restoring_force_prime(xi, parameters.alpha)
    fpp = restoring_force_second(xi, parameters.alpha)
    z = complex_linear_response(xi, parameters)
    second_denominator = (
        kappa - 4.0 * parameters.omega**2
        + 2j * parameters.delta * parameters.omega
    )
    return -fpp / 4.0 * (
        abs(z) ** 2 / kappa
        + np.real(z**2 * np.exp(2j * parameters.omega * time_array) / second_denominator)
    )


def central_cubic_correction(
    time: ArrayLike,
    parameters: Parameters,
) -> NDArray[np.float64]:
    time_array = np.asarray(time, dtype=float)
    kappa = 1.0 - parameters.alpha
    z = 1.0 / (
        kappa - parameters.omega**2
        + 1j * parameters.delta * parameters.omega
    )
    third_denominator = (
        kappa - 9.0 * parameters.omega**2
        + 3j * parameters.delta * parameters.omega
    )
    first_harmonic = -3.0 * parameters.alpha * abs(z) ** 2 / 8.0 * np.real(
        z**2 * np.exp(1j * parameters.omega * time_array)
    )
    third_harmonic = -parameters.alpha / 8.0 * np.real(
        z**3 * np.exp(3j * parameters.omega * time_array) / third_denominator
    )
    return first_harmonic + third_harmonic


def first_order_approximation(
    time: ArrayLike,
    beta: float,
    xi: float,
    parameters: Parameters,
) -> NDArray[np.float64]:
    return xi + beta * first_order_correction(time, xi, parameters)


def second_order_approximation(
    time: ArrayLike,
    beta: float,
    xi: float,
    parameters: Parameters,
) -> NDArray[np.float64]:
    return (
        xi
        + beta * first_order_correction(time, xi, parameters)
        + beta**2 * second_order_correction(time, xi, parameters)
    )


def central_cubic_approximation(
    time: ArrayLike,
    beta: float,
    parameters: Parameters,
) -> NDArray[np.float64]:
    return (
        beta * first_order_correction(time, 0.0, parameters)
        + beta**3 * central_cubic_correction(time, parameters)
    )
