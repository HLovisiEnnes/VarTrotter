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

from .utils import get_commutators_from_list


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

    def is_subalgebra(self, rtol: float = 1e-6) -> bool:
        """
        Checks if the pulses in the pulses_gen list form a subalgebra of su(d).

        Args:
            rtol (float): Relative tolerance for the linear system solution.
                Default is 1e-6.

        Returns:
            bool: True if the pulses form a subalgebra, False otherwise.
        """
        # Vectorize the basis
        basis = np.column_stack([A.reshape(-1) for A in self.pulses_gen])

        for comm in self.commutators():
            # Solves the linear system problem to see if the commutator is in the
            # span of the hermirtian basis.
            coeffs, *_ = np.linalg.lstsq(
                basis,
                comm.reshape(-1),
                rcond=None,
            )

            residual = np.linalg.norm(basis @ coeffs - comm.reshape(-1))

            # We take the square of the max_norm because the commutator norm is
            # quadratic on the pulses' norms
            if residual > 2 * rtol * self._max_norm**2:
                return False

        return True
