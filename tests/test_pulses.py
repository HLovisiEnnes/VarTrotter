"""
----------------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
----------------------------------------------------------------------------------------
Tests for the pulses module.
----------------------------------------------------------------------------------------
"""

import numpy as np
import pytest

from vartrotter.pulses import Pulses


@pytest.fixture
def closed_pulses():
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    return Pulses([X, Y, Z])


@pytest.fixture
def non_closed_pulses():
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    return Pulses([X, Z])


def test_len(closed_pulses, non_closed_pulses):
    assert len(closed_pulses) == 3
    assert len(non_closed_pulses) == 2


def test_iter(closed_pulses, non_closed_pulses):
    assert list(closed_pulses) == closed_pulses.pulses_gen
    assert list(non_closed_pulses) == non_closed_pulses.pulses_gen


def test_getitem(closed_pulses, non_closed_pulses):
    assert closed_pulses[0] is closed_pulses.pulses_gen[0]
    assert non_closed_pulses[1] is non_closed_pulses.pulses_gen[1]


def test_commutators(closed_pulses):
    expected = [
        np.array([[-2, 0], [0, 2]], dtype=complex),
        np.array([[0, -2j], [2j, 0]], dtype=complex),
        np.array([[0, -2], [-2, 0]], dtype=complex),
    ]

    for comm, exp in zip(closed_pulses.commutators(), expected):
        np.testing.assert_allclose(comm, exp)


def test_is_subalgebra_closed(closed_pulses, non_closed_pulses):
    assert closed_pulses.is_subalgebra()
    assert not non_closed_pulses.is_subalgebra()
