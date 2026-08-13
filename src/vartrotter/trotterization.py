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


class Trotterization:
    """
    Trotterization class, for both fixed-weight and variable-weight solutions.

    Args:
        pulses (Pulses): A Pulses object containing the pulse generators.
        N (int): The number of Trotter steps.
        target (np.ndarray): The target matrix to be approximated.
        coeffs (Literal["R", "Z"]): Whether to use real or integer coefficients for
            the fixed-weight solution. Default is "R" for real coefficients.

    Public attributes:
        pulses: Pulses - A Pulses object containing the pulse generators.
        N: int - The number of Trotter steps.
        target: np.ndarray - The target matrix to be approximated.
        coeffs: Literal["R", "Z"] - Whether to use real or integer coefficients for
            the fixed-weight solution.
        norm_target: np.ndarray - The normalized target matrix (target / N).

    Public methods:
        compute_weights(H: np.ndarray | None = None) -> np.ndarray: Computes the
            coefficients for the fixed-weight algorithm by row-reduction.
        compute_drift(weights: np.ndarray) -> np.ndarray: Computes the drift of the
            trotterization.
        compute_actual_error_fixed_weights(return_appx: bool = False) -> float:
            Computes the actual error of the fixed-weight trotterization solution.
        compute_crude_error_bound_fixed_weights() -> float: Computes the crude triangle
            inequality bound for the fixed-weight solution.
        compute_usual_error_bound_fixed_weights() -> float: Computes the usual error
            bound for the fixed-weight solution.
        compute_fourier_error_bound_fixed_weights() -> float: Computes the Fourier-type
            error bound for the fixed-weight solution.
        fixed_weights() -> tuple[np.ndarray, np.ndarray, dict]: Computes the
            fixed-weight trotterization solution and associated errors.
        compute_schedule() -> tuple[list[np.ndarray], np.ndarray]: Computes the
            variable-weight schedule and accumulated drift R.
        variable_weights() -> tuple[list[np.ndarray], np.ndarray, dict]: Computes the
            variable-weight trotterization solution and associated errors.
    """

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
        # Compute the eigenvalues and eigenbasis of the target matrix
        self._evals, self._V = np.linalg.eigh(self.target)
        self._omega = self._evals[:, None] - self._evals[None, :]

    def compute_weights(self, H: np.ndarray | None = None) -> np.ndarray:
        """
        Computes the coefficients for the fixed-weight algorithm by row-reduction.
        If coefficients are `R`, this are just the output of row-reduction; if they
        are `Z`, they are rounded to the nearest integer.

        Args:
            H (np.ndarray | None): The target matrix. If None, uses the normalized
                target matrix.

        Returns:
            np.ndarray: Coefficients for the fixed-weight algorithm.
        """
        basis = np.column_stack([P.reshape(-1) for P in self.pulses])
        if H is None:
            H = self.norm_target

        H = H.reshape(-1)

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

        return rec - self.norm_target

    """
    Fixed-weights solution
    """

    def compute_actual_error_fixed_weights(self, return_appx: bool = False) -> float:
        """
        Computes the actual error of the fixed-weight trotterization solution.

        Args:
            return_appx (bool): If True, returns the approximate unitary as well.
                Default is False.
        Returns:
            float: The actual error of the fixed-weight trotterization solution.
        """
        weights = self.compute_weights()

        T = np.eye(self.pulses[0].shape[0], dtype=complex)

        for i, w in enumerate(weights):
            T @= linalg.expm(1j * w * self.pulses[i])

        appx = np.linalg.matrix_power(T, self.N)

        if return_appx:
            return appx, np.linalg.norm(linalg.expm(1j * self.target) - appx, ord=2)

        else:
            return np.linalg.norm(linalg.expm(1j * self.target) - appx, ord=2)

    def compute_crude_error_bound_fixed_weights(self):
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

    def compute_usual_error_bound_fixed_weights(self):
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

    def compute_fourier_error_bound_fixed_weights(self):
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

        # Numpy uses normalized sinc, so we need to multiply by 2pi to get
        # the unnormalized version
        factor = (
            np.exp(-0.5j * (self.N - 1) / self.N * self._omega)
            * self.N
            * np.sinc(self._omega / (2 * np.pi))
            / np.sinc(self._omega / (2 * np.pi * self.N))
        )

        weighted_pulse = [weights[i] * self.pulses[i] for i in range(len(weights))]
        comms = get_commutators_from_list(weighted_pulse)
        squared_term = 0
        for comm in comms:
            squared_term += comm

        # Move everything to the eigenbasis of the target matrix
        drift = adjoint(self._V.conj().T, drift)
        squared_term = adjoint(self._V.conj().T, squared_term)

        return np.linalg.norm(factor * drift, ord=2) + 1 / 2 * np.linalg.norm(
            factor * squared_term, ord=2
        )

    def fixed_weights(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Computes the fixed-weight trotterization solution and the associated errors.

        Returns:
            tuple: A tuple containing the weights and a dictionary with the actual
                error, crude estimate, usual error bound, and Fourier error bound.
        """
        appx, actual_err = self.compute_actual_error_fixed_weights(return_appx=True)
        return (
            self.compute_weights(),
            appx,
            {
                "Actual error": actual_err,
                "Crude estimate": self.compute_crude_error_bound_fixed_weights(),
                "Usual bound": self.compute_usual_error_bound_fixed_weights(),
                "Fourier bound": self.compute_fourier_error_bound_fixed_weights(),
            },
        )

    """
    Variable-weights solution
    """

    def compute_schedule(self) -> tuple[list[np.ndarray], np.ndarray]:
        schedule = []

        R = np.zeros(self.target.shape, dtype=complex)
        for n in range(1, self.N + 1):
            phase = np.exp(1j * (n - 1) * self._omega / self.N)

            # Desired drift in the H eigenbasis:
            # Delta^(n) ~= -Ad_{S^(n-1)} R^(n-1)
            desired_delta_eig = -phase * R

            # Transform back to the physical basis because compute_weights()
            # uses the original pulses.
            desired_delta = adjoint(self._V, desired_delta_eig)

            # Next weights are just the weihts around alpha + the weights to kill
            # the accumulated error
            next_weights = self.compute_weights(H=self.norm_target + desired_delta)
            schedule.append(next_weights)

            # Assume everything in the eigenbasis
            delta = adjoint(self._V.conj().T, self.compute_drift(next_weights))

            weighted_pulse = [
                next_weights[i] * self.pulses[i] for i in range(len(next_weights))
            ]
            comms = get_commutators_from_list(weighted_pulse)
            squared_term = 0
            for comm in comms:
                squared_term += comm
            squared_term = (
                1 / 2 * adjoint(self._V.conj().T, squared_term)
                - 1 / (2 * self.N) * 1j * self._omega * delta
            )

            R += np.conj(phase) * (delta + squared_term)

            # Debug term
            assert np.allclose(R, R.conj().T)
            assert np.allclose(delta, delta.conj().T)

        return schedule, R

    def variable_weights(self) -> tuple[list[np.ndarray], np.ndarray, dict]:
        schedule, R = self.compute_schedule()

        for n, weights in enumerate(schedule):
            T = np.eye(self.pulses[0].shape[0], dtype=complex)
            for i, w in enumerate(weights):
                T @= linalg.expm(1j * w * self.pulses[i])
            if n == 0:
                appx = T
            else:
                appx @= T

        actual_err = np.linalg.norm(linalg.expm(1j * self.target) - appx, ord=2)

        return (
            schedule,
            appx,
            {
                "Actual error": actual_err,
                "Predict R_int": np.linalg.norm(R, ord=2),
            },
        )
