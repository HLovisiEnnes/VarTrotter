"""
----------------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
----------------------------------------------------------------------------------------
The main trotterization class.
----------------------------------------------------------------------------------------
Utils:
    FixedWeightTrotterization
----------------------------------------------------------------------------------------
"""

from typing import Literal

import numpy as np
from scipy import linalg

from .pulses import Pulses
from .utils import adjoint, get_commutators_from_list


class FixedWeightTrotterization:
    def __init__(
        self,
        pulses: Pulses,
        N: int,
        target: np.ndarray,
        coeffs: Literal["R", "Z"] = "R",
    ):
        self.pulses = pulses
        self.N = N
        self.target = target
        self.coeffs = coeffs
        self.norm_target = target / N

    def compute_weights(self) -> np.ndarray:
        """
        Computes the coefficients for the fixed-weight algorithm by row-reduction.
        If coefficients are `R`, this are just the output of row-reduction; if they
        are `Z`, they are rounded to the nearest integer.

        Returns:
            np.ndarray: Coefficients for the fixed-weight algorithm.
        """
        basis = np.column_stack([P.reshape(-1) for P in self.pulses])
        H = self.norm_target.reshape(-1)

        # Solves the system as real, since we want a real linear combination only
        real_basis = np.vstack([basis.real, basis.imag])
        real_target = np.concatenate([H.real, H.imag])

        weights, *_ = np.linalg.lstsq(real_basis, real_target, rcond=None)

        if self.coeffs == "Z":
            weights = np.round(weights)

        return weights

    def compute_drift(self, weights: np.ndarray) -> np.ndarray:
        """
        Computes the drift of the trotterization.

        Args:
            weights (np.ndarray): Coefficients for the fixed-weight algorithm.
        Returns:
            np.ndarray: The drift of the trotterization.
        """
        rec = 0
        for i, w in enumerate(weights):
            rec += w * self.pulses[i]

        return self.norm_target - rec

    def compute_actual_error(self) -> float:
        """
        Computes the actual error of the fixed-weight trotterization solution.
        """
        weights = self.compute_weights()

        T = np.eye(self.pulses[0].shape[0], dtype=complex)

        for i, w in enumerate(weights):
            T @= linalg.expm(1j * w * self.pulses[i])

        return np.linalg.norm(
            linalg.expm(1j * self.target) - np.linalg.matrix_power(T, self.N), ord=2
        )

    def compute_crude_error_bound(self):
        """
        Computes the crude triangle inequality bound.
        """
        weights = self.compute_weights()

        # In the case where coef = `R`, we assume exact infinitesimal synthesis, so
        # in the fixed-weight solution, the drift is zero
        if self.coeffs == "R":
            drift = np.zeros(self.target.shape, dtype=complex)
        else:
            drift = self.compute_drift(weights)

        # Recall that the squared term is the sum of commutators norms, which is
        # quadratic in the pulses' norms
        squared_term = 0
        j = 0
        for i, j, comm in self.pulses.commutators(indices=True):
            squared_term += np.linalg.norm(comm, ord=2) * abs(weights[i] * weights[j])

        return self.N * (np.linalg.norm(drift, ord=2) + 1 / 2 * squared_term)

    def compute_usual_error_bound(self):
        """
        Computes the usual error bound, where we do not apply the triangle inequality.
        """
        weights = self.compute_weights()

        # In the case where coef = `R`, we assume exact infinitesimal synthesis, so
        # in the fixed-weight solution, the drift is zero
        if self.coeffs == "R":
            drift = np.zeros(self.target.shape, dtype=complex)
        else:
            drift = self.compute_drift(weights)

        weighted_pulse = [weights[i] * self.pulses[i] for i in range(len(weights))]
        comms = get_commutators_from_list(weighted_pulse)
        squared_term = 0
        for comm in comms:
            squared_term += comm

        return self.N * (
            np.linalg.norm(drift, ord=2) + 1 / 2 * np.linalg.norm(squared_term, ord=2)
        )

    def compute_fourier_error_bound(self):
        """
        Computes the Fourier-type error bound.
        """
        weights = self.compute_weights()

        # In the case where coef = `R`, we assume exact infinitesimal synthesis, so
        # in the fixed-weight solution, the drift is zero
        if self.coeffs == "R":
            drift = np.zeros(self.target.shape, dtype=complex)
        else:
            drift = self.compute_drift(weights)

        # Compute the eigenvalues and eigenbasis of the target matrix
        evals, V = np.linalg.eigh(self.target)
        omega = evals[:, None] - evals[None, :]
        # Numpy uses normalized sinc, so we need to multiply by 2pi to get
        # the unnormalized version
        factor = (
            np.exp(-0.5j * (self.N - 1) / self.N * omega)
            * self.N
            * np.sinc(omega / (2 * np.pi))
            / np.sinc(omega / (2 * np.pi * self.N))
        )

        weighted_pulse = [weights[i] * self.pulses[i] for i in range(len(weights))]
        comms = get_commutators_from_list(weighted_pulse)
        squared_term = 0
        for comm in comms:
            squared_term += comm

        # Move everything to the eigenbasis of the target matrix
        drift = adjoint(V.conj().T, drift)
        squared_term = adjoint(V.conj().T, squared_term)

        return np.linalg.norm(factor * drift, ord=2) + 1 / 2 * np.linalg.norm(
            factor * squared_term, ord=2
        )
