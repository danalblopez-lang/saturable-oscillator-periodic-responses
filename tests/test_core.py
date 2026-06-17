from __future__ import annotations

import numpy as np

from saturable_oscillator.model import Parameters, zero_forcing_states
from saturable_oscillator.shooting import solve_periodic_orbit


def test_constant_states_are_periodic_at_zero_forcing():
    parameters = Parameters(alpha=2.0, delta=0.35, omega=1.0)
    for xi in zero_forcing_states(parameters.alpha):
        result = solve_periodic_orbit(
            np.array([xi, 0.0]),
            parameters,
            beta=0.0,
        )
        assert result.residual_norm < 1.0e-9
        assert np.allclose(result.initial_state, [xi, 0.0], atol=1.0e-8)


def test_liouville_identity_at_representative_orbit():
    parameters = Parameters(alpha=2.0, delta=0.35, omega=1.0)
    xi = zero_forcing_states(parameters.alpha)[2]
    state = np.array([xi, 0.0])
    for beta in np.linspace(0.0, 0.20, 41):
        result = solve_periodic_orbit(state, parameters, float(beta))
        state = result.initial_state
    assert result.liouville_defect < 1.0e-8
