"""
----------------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
----------------------------------------------------------------------------------------
Tests for the trotterization module.
----------------------------------------------------------------------------------------
"""

import numpy as np

from vartrotter.pulses import Pulses
from vartrotter.trotterization import FixedWeightTrotterization


def test_errors():
    X = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=complex,
    )

    Z = np.array(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=complex,
    )

    g = 1.0

    N = 10
    H1 = np.pi * Z + g * X
    H2 = np.pi * Z - g * X

    pulses = [H1, H2]
    weights = np.array([0.5, 0.5])

    H = weights[0] * H1 + weights[1] * H2

    pulses = Pulses(pulses)
    trotter = FixedWeightTrotterization(pulses, N, H)

    # Actual and geoemtric should be close to zero, the other ones should
    # be at least 0.15

    assert trotter.compute_actual_error() < 0.01
    assert trotter.compute_fourier_error_bound() < 0.01
    assert trotter.compute_crude_error_bound() >= 0.15
    assert trotter.compute_usual_error_bound() >= 0.15
