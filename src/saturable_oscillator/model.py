from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Parameters:
    alpha: float = 2.0
    delta: float = 0.35
    omega: float = 1.0

    @property
    def period(self) -> float:
        return 2.0 * np.pi / self.omega


def restoring_force(u: ArrayLike, alpha: float) -> NDArray[np.float64] | float:
    u_arr = np.asarray(u)
    out = u_arr - alpha * u_arr / np.sqrt(1.0 + u_arr * u_arr)
    return float(out) if out.ndim == 0 else out


def restoring_force_prime(u: ArrayLike, alpha: float) -> NDArray[np.float64] | float:
    u_arr = np.asarray(u)
    out = 1.0 - alpha / (1.0 + u_arr * u_arr) ** 1.5
    return float(out) if out.ndim == 0 else out


def restoring_force_second(u: ArrayLike, alpha: float) -> NDArray[np.float64] | float:
    u_arr = np.asarray(u)
    out = 3.0 * alpha * u_arr / (1.0 + u_arr * u_arr) ** 2.5
    return float(out) if out.ndim == 0 else out


def critical_point(alpha: float) -> float:
    if alpha <= 1.0:
        raise ValueError("critical_point requires alpha > 1")
    return float(np.sqrt(alpha ** (2.0 / 3.0) - 1.0))


def critical_half_width(alpha: float) -> float:
    if alpha <= 1.0:
        raise ValueError("critical_half_width requires alpha > 1")
    return float((alpha ** (2.0 / 3.0) - 1.0) ** 1.5)


def zero_forcing_states(alpha: float) -> tuple[float, float, float]:
    if alpha <= 1.0:
        raise ValueError("zero_forcing_states requires alpha > 1")
    exterior = float(np.sqrt(alpha * alpha - 1.0))
    return -exterior, 0.0, exterior


def global_second_derivative_bound(alpha: float) -> float:
    return float(48.0 * alpha / (25.0 * np.sqrt(5.0)))
