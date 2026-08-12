import matplotlib.pyplot as plt
import numpy as np

from vartrotter.pulses import Pulses
from vartrotter.trotterization import FixedWeightTrotterization
from vartrotter.utils import random_su_d

"""
Defines the seed
"""
seed = 42

"""
Defines some useful pulses
"""
X = np.array([[0, 1], [1, 0]], dtype=complex)

Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

Z = np.array([[1, 0], [0, -1]], dtype=complex)

closed_pulses = Pulses([X, Y, Z])


"""
Plots the four kinds of estimated errors
"""
errs_actual = []
errs_crude = []
errs_usual = []
errs_geometric = []
Ns = list(range(10, 50))
M = random_su_d(2, seed=seed)

for N in Ns:
    trotter = FixedWeightTrotterization(closed_pulses, N, M)
    errs_actual.append(trotter.compute_actual_error())
    errs_crude.append(trotter.compute_crude_error_bound())
    errs_usual.append(trotter.compute_usual_error_bound())
    errs_geometric.append(trotter.compute_fourier_error_bound())

plt.plot(Ns, errs_actual, "--", c="y", label="Actual Error")
plt.plot(Ns, errs_crude, c="xkcd:dark orange", label="Crude Bound")
plt.plot(Ns, errs_usual, c="xkcd:brick red", label="Usual Bound")
plt.plot(Ns, errs_geometric, c="xkcd:navy", label="Fourier Bound")
plt.legend()
plt.xlabel(r"Number of Trotter Steps ($N$)")
plt.ylabel("Error")
plt.savefig("figures/errors_fixed_weights_drift_free.pdf")

"""
Plots the actual error vs the Fourier error bound for a randomly generated Hamiltonians
"""
ratios = []
actual_errors = []
fourier_errors = []

N = 50

for seed in range(100):
    H = random_su_d(2, seed=seed)

    trot = FixedWeightTrotterization(closed_pulses, N, H)

    actual = trot.compute_actual_error()
    fourier = trot.compute_fourier_error_bound()

    actual_errors.append(actual)
    fourier_errors.append(fourier)
    ratios.append(fourier / actual)

actual_errors = np.array(actual_errors)
fourier_errors = np.array(fourier_errors)
ratios = np.array(ratios)

# Scatter plot: Fourier prediction vs actual error
plt.scatter(actual_errors, fourier_errors)

m = max(actual_errors.max(), fourier_errors.max())
plt.plot([0, m], [0, m], "--")

plt.xlabel("Actual error")
plt.ylabel("Fourier error")
plt.savefig("figures/actual_vs_fourier.pdf")
