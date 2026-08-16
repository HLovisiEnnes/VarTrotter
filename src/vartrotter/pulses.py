"""
----------------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
----------------------------------------------------------------------------------------
The main pulses class.
----------------------------------------------------------------------------------------
Utils:
    Pulses
----------------------------------------------------------------------------------------
"""

import numpy as np

from .utils import get_commutators_from_list, real_vec


class Pulses:
    """
    A class representing a set of pulse generators in su(d).

    Args:
        pulses_gen (list[np.ndarray]): List of matrices representing the pulses.

    Public attributes:
        pulses_gen: list[np.ndarray] - List of matrices representing the pulses.

    Public methods:
        len() -> int: Returns the number of pulses in the pulses_gen list.
        iter() -> generator: Returns an iterator over the pulses in the pulses_gen list.
        getitem(index: int) -> np.ndarray: Returns the pulse at the specified index in
            the pulses_gen list.
        commutators() -> generator: A generator object that yields the commutators of
            the pulses in the pulses_gen list.
        is_subalgebra(rtol: float = 1e-6) -> bool: Checks if the pulses in the
            pulses_gen list form a subalgebra of su(d).
    """

    def __init__(self, pulses_gen: list[np.ndarray]) -> None:
        self.pulses_gen = pulses_gen
        self._max_norm = max(np.linalg.norm(pulse, ord=2) for pulse in pulses_gen)

    def __len__(self):
        return len(self.pulses_gen)

    def __iter__(self):
        return iter(self.pulses_gen)

    def __getitem__(self, index):
        return self.pulses_gen[index]

    def commutators(self, indices: bool = False):
        """
        A generator object that yields the commutators of the pulses in the pulses_gen
            list.
        """
        return get_commutators_from_list(self.pulses_gen, indices=indices)

    def is_subalgebra(self, atol: float = 1e-10, rtol: float = 1e-6) -> bool:
        """
        Checks if the pulses in the pulses_gen list form a subalgebra of su(d).

        Args:
            atol (float): Absolute tolerance for the linear system solution.
                Default is 1e-10.
            rtol (float): Relative tolerance for the linear system solution.
                Default is 1e-6.

        Returns:
            bool: True if the pulses form a subalgebra, False otherwise.
        """
        # Vectorize the basis
        # Real-vectorize the basis, because su(d) is a real vector space.
        basis = np.column_stack([real_vec(A) for A in self.pulses_gen])

        for comm in self.commutators():
            comm_vec = real_vec(comm)

            # Solve over R to check whether the commutator lies in the real span
            # of the Hermitian basis.
            coeffs, *_ = np.linalg.lstsq(
                basis,
                comm_vec,
                rcond=None,
            )

            residual = np.linalg.norm(basis @ coeffs - comm_vec)

            # We take the square of the max_norm because the commutator norm is
            # quadratic on the pulses' norms
            if residual > atol + 2 * rtol * self._max_norm**2:
                return False

        return True
