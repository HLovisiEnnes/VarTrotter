"""
----------------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
----------------------------------------------------------------------------------------
Tests for the trotterization module.
----------------------------------------------------------------------------------------
"""

import numpy as np
import pytest

from vartrotter.pulses import Pulses
from vartrotter.trotterization import Trotterization


@pytest.fixture
def pulses_non_closed():
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    g = 1.0

    H1 = np.pi * Z + g * X
    H2 = np.pi * Z - g * X

    return Pulses([H1, H2])


@pytest.fixture
def pulses_closed():
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    return Pulses([X, Y, Z])


def test_errors_fixed(pulses_non_closed):

    N = 10
    H = 0.5 * pulses_non_closed[0] + 0.5 * pulses_non_closed[1]

    trotter = Trotterization(pulses_non_closed, N, H)

    # Actual and geoemtric should be close to zero, the other ones should
    # be at least 0.15
    assert trotter.compute_actual_error_fixed_weights() < 0.01
    assert trotter.compute_fourier_error_bound_fixed_weights() < 0.01
    assert trotter.compute_crude_error_bound_fixed_weights() >= 0.15
    assert trotter.compute_usual_error_bound_fixed_weights() >= 0.15


def test_errors_var(pulses_closed):
    M = np.array(
        [
            [1.54820991 + 0.0j, 2.33776155 - 1.41687029j],
            [2.33776155 + 1.41687029j, -1.54820991 + 0.0j],
        ]
    )
    N = 40
    trotter = Trotterization(pulses_closed, N, M)

    assert (
        trotter.compute_actual_error_fixed_weights()
        >= trotter.variable_weights()[2]["Actual error"]
    )
