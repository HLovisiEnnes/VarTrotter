"""
----------------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
----------------------------------------------------------------------------------------
Tests for the utils module.
----------------------------------------------------------------------------------------
"""

import numpy as np

from vartrotter.utils import (
    adjoint,
    get_commutator,
    get_commutators_from_list,
    random_su_d,
)


def test_get_commutator():
    A = np.array([[0, 1], [-1, 0]], dtype=complex)
    B = np.array([[1, 0], [0, -1]], dtype=complex)
    expected = 1j * (A @ B - B @ A)
    assert np.array_equal(get_commutator(A, B), expected)


def test_get_commutators_from_list():
    matrices = [
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]

    expected = [
        np.array([[-2, 0], [0, 2]], dtype=complex),
        np.array([[0, -2j], [2j, 0]], dtype=complex),
        np.array([[0, -2], [-2, 0]], dtype=complex),
    ]

    commutators = list(get_commutators_from_list(matrices))

    assert len(commutators) == len(expected)

    for comm, exp in zip(commutators, expected):
        np.testing.assert_allclose(comm, exp)


def test_adjoint():
    X = random_su_d(2, seed=42)
    U = np.array([[0, 1], [-1, 0]], dtype=complex)
    assert np.array_equal(adjoint(U, X), U @ X @ U.conj().T)
