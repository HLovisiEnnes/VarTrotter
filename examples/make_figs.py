import matplotlib.pyplot as plt
import numpy as np

from vartrotter.pulses import Pulses
from vartrotter.trotterization import Trotterization
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

# Full controllability
closed_pulses = Pulses([X, Y, Z])

# Mimics gate synethsis
norm = 100
small_pulses = [X / norm, Y / norm, Z / norm]
small_pulses = Pulses(small_pulses)

M = random_su_d(2, seed=seed)
print("Target:", M)

"""
Plots the four kinds of estimated errors
"""
errs_actual = []
errs_crude = []
errs_usual = []
errs_geometric = []
Ns = list(range(10, 50))


for N in Ns:
    trotter = Trotterization(closed_pulses, N, M)
    errs_actual.append(trotter.compute_actual_error_fixed_weights())
    errs_crude.append(trotter.compute_crude_error_bound_fixed_weights())
    errs_usual.append(trotter.compute_usual_error_bound_fixed_weights())
    errs_geometric.append(trotter.compute_fourier_error_bound_fixed_weights())

plt.plot(Ns, errs_actual, "--", c="y", label="Actual Error")
plt.plot(Ns, errs_crude, c="xkcd:dark orange", label="Crude Bound")
plt.plot(Ns, errs_usual, c="xkcd:brick red", label="Usual Bound")
plt.plot(Ns, errs_geometric, c="xkcd:navy", label="Fourier Bound")
plt.legend()
plt.xlabel(r"$N$")
plt.ylabel(r"$\varepsilon$")
plt.savefig("figures/errors_fixed_weights_drift_free.pdf")
plt.show()


"""
Plots the four kinds of estimated errors when we have imperfect synthesis
"""
errs_actual = []
errs_crude = []
errs_usual = []
errs_geometric = []
Ns = list(range(10, 50))


for N in Ns:
    trotter = Trotterization(small_pulses, N, M, coeffs="Z")
    errs_actual.append(trotter.compute_actual_error_fixed_weights())
    errs_crude.append(trotter.compute_crude_error_bound_fixed_weights())
    errs_usual.append(trotter.compute_usual_error_bound_fixed_weights())
    errs_geometric.append(trotter.compute_fourier_error_bound_fixed_weights())

plt.plot(Ns, errs_actual, "--", c="y", label="Actual Error")
plt.plot(Ns, errs_crude, c="xkcd:dark orange", label="Crude Bound")
plt.plot(Ns, errs_usual, c="xkcd:brick red", label="Usual Bound")
plt.plot(Ns, errs_geometric, c="xkcd:navy", label="Fourier Bound")
plt.legend()
plt.xlabel(r"$N$")
plt.ylabel(r"$\varepsilon$")
plt.savefig("figures/errors_fixed_weights_with_drift.pdf")
plt.show()

"""
Plots the actual error vs the Fourier error bound for a randomly generated Hamiltonians
"""
ratios = []
actual_errors = []
fourier_errors = []

N = 50

for seed in range(100):
    H = random_su_d(2, seed=seed)

    trot = Trotterization(closed_pulses, N, H)

    actual = trot.compute_actual_error_fixed_weights()
    fourier = trot.compute_fourier_error_bound_fixed_weights()

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
plt.show()


"""
Plots the variable vs fixed weights error scaling for a randomly generated Hamiltonian
"""
errs_fixed = []
errs_var = []
Ns = list(range(10, 80))


for N in Ns:
    trotter = Trotterization(closed_pulses, N, M)
    errs_fixed.append(trotter.compute_actual_error_fixed_weights())
    errs_var.append(trotter.variable_weights()[2]["Actual error"])


slope_fixed, _ = np.polyfit(np.log(Ns), np.log(errs_fixed), 1)
slope_var, _ = np.polyfit(np.log(Ns), np.log(errs_var), 1)


plt.loglog(Ns, errs_fixed, label=rf"Fixed weights ($p={slope_fixed:.2f}$)")
plt.loglog(Ns, errs_var, label=rf"Scheduled weights ($p={slope_var:.2f}$)")


# Reference scalings
plt.loglog(
    Ns,
    errs_fixed[0] * (Ns[0] / np.array(Ns)),
    ":",
    label=r"$N^{-1}$",
)

plt.loglog(
    Ns,
    errs_var[0] * (Ns[0] / np.array(Ns)) ** 2,
    ":",
    label=r"$N^{-2}$",
)

plt.legend()

plt.xlabel(r"$\log(N)$")
plt.ylabel(r"$\log(\epsilon)$")
plt.savefig("figures/var_vs_fixed_r.pdf")
plt.show()


"""
Plots the variable vs fixed weights error scaling for a randomly generated Hamiltonian
and integer coefficients, which mimics the case of imperfect synthesis
"""

errs_fixed = []
errs_var = []
Ns = list(range(10, 80))

for N in Ns:
    trotter = Trotterization(small_pulses, N, M, coeffs="Z")
    errs_fixed.append(trotter.compute_actual_error_fixed_weights())
    errs_var.append(trotter.variable_weights()[2]["Actual error"])


slope_fixed, _ = np.polyfit(np.log(Ns), np.log(errs_fixed), 1)
slope_var, _ = np.polyfit(np.log(Ns), np.log(errs_var), 1)


plt.loglog(Ns, errs_fixed, label=rf"Fixed weights ($p={slope_fixed:.2f}$)")
plt.loglog(Ns, errs_var, label=rf"Scheduled weights ($p={slope_var:.2f}$)")


# Reference scalings
plt.loglog(
    Ns,
    errs_fixed[0] * (Ns[0] / np.array(Ns)),
    ":",
    label=r"$N^{-1}$",
)

plt.loglog(
    Ns,
    errs_var[0] * (Ns[0] / np.array(Ns)) ** 2,
    ":",
    label=r"$N^{-2}$",
)

plt.legend()

plt.xlabel(r"$\log(N)$")
plt.ylabel(r"$\log(\varepsilon)$")
plt.savefig("figures/var_vs_fixed_z.pdf")
plt.show()
