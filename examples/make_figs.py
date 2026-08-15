import argparse

import matplotlib.pyplot as plt
import numpy as np

from vartrotter.pulses import Pulses
from vartrotter.trotterization import Trotterization
from vartrotter.utils import random_su_d

"""
Defines constants
"""

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_min", type=int, default=10)
parser.add_argument("--n_max", type=int, default=100)
parser.add_argument("--norm", type=int, default=50)
parser.add_argument("--save", type=bool, default=False)
parser.add_argument("--ext", type=str, default="")
args = parser.parse_args()

seed = args.seed
Ns = list(range(args.n_min, args.n_max))
norm = args.norm
save = args.save
file_ext = args.ext

"""
Defines some useful pulses
"""
X = np.array([[0, 1], [1, 0]], dtype=complex)

Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Full controllability
closed_pulses = Pulses([X, Y, Z])

# Mimics gate synethsis
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
if save:
    plt.savefig("figures/errors_fixed_weights_drift_free" + file_ext + ".pdf")
plt.show()


"""
Plots the four kinds of estimated errors when we have imperfect synthesis
"""
errs_actual = []
errs_crude = []
errs_usual = []
errs_geometric = []


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
if save:
    plt.savefig("figures/errors_fixed_weights_with_drift" + file_ext + ".pdf")
plt.show()

"""
Plots the actual error vs the Fourier error bound for a randomly generated Hamiltonians
"""
ratios = []
actual_errors = []
fourier_errors = []

N = 50

for s in range(100):
    H = random_su_d(2, seed=s)

    trotter = Trotterization(closed_pulses, N, H)

    actual = trotter.compute_actual_error_fixed_weights()
    fourier = trotter.compute_fourier_error_bound_fixed_weights()

    actual_errors.append(actual)
    fourier_errors.append(fourier)
    ratios.append(fourier / actual)

actual_errors = np.array(actual_errors)
fourier_errors = np.array(fourier_errors)
ratios = np.array(ratios)

# Scatter plot: Fourier prediction vs actual error
plt.scatter(actual_errors, fourier_errors, c="xkcd:navy")

m = max(actual_errors.max(), fourier_errors.max())
plt.plot([0, m], [0, m], "--", c="xkcd:navy")

plt.xlabel("Actual error")
plt.ylabel("Fourier error")
if save:
    plt.savefig("figures/actual_vs_fourier" + file_ext + ".pdf")
plt.show()


"""
Plots the variable vs fixed weights error scaling for a randomly generated Hamiltonian
"""
errs_fixed = []
errs_var = []

for N in Ns:
    trotter = Trotterization(closed_pulses, N, M)
    errs_fixed.append(trotter.compute_actual_error_fixed_weights())
    errs_var.append(trotter.variable_weights()[2]["Actual error"])


slope_fixed, _ = np.polyfit(np.log(Ns), np.log(errs_fixed), 1)
slope_var, _ = np.polyfit(np.log(Ns), np.log(errs_var), 1)


plt.loglog(
    Ns, errs_fixed, label=rf"Fixed weights ($p={slope_fixed:.2f}$)", c="xkcd:brick red"
)
plt.loglog(
    Ns, errs_var, label=rf"Scheduled weights ($p={slope_var:.2f}$)", c="xkcd:navy"
)


# Reference scalings
plt.loglog(
    Ns,
    errs_fixed[0] * (Ns[0] / np.array(Ns)),
    ":",
    label=r"$N^{-1}$",
    c="xkcd:brick red",
)

plt.loglog(
    Ns, errs_var[0] * (Ns[0] / np.array(Ns)) ** 2, ":", label=r"$N^{-2}$", c="xkcd:navy"
)

plt.legend()

plt.xlabel(r"$\log(N)$")
plt.ylabel(r"$\log(\epsilon)$")
if save:
    plt.savefig("figures/var_vs_fixed_r" + file_ext + ".pdf")
plt.show()


"""
Plots the variable vs fixed weights error scaling for a randomly generated Hamiltonian
and integer coefficients, which mimics the case of imperfect synthesis
"""
errs_fixed = []
errs_var = []

for N in Ns:
    trotter = Trotterization(small_pulses, N, M, coeffs="Z")
    errs_fixed.append(trotter.compute_actual_error_fixed_weights())
    errs_var.append(trotter.variable_weights()[2]["Actual error"])


slope_fixed, _ = np.polyfit(np.log(Ns), np.log(errs_fixed), 1)
slope_var, _ = np.polyfit(np.log(Ns), np.log(errs_var), 1)


plt.loglog(
    Ns, errs_fixed, label=rf"Fixed weights ($p={slope_fixed:.2f}$)", c="xkcd:brick red"
)
plt.loglog(
    Ns, errs_var, label=rf"Scheduled weights ($p={slope_var:.2f}$)", c="xkcd:navy"
)


# Reference scalings
plt.loglog(
    Ns,
    errs_fixed[0] * (Ns[0] / np.array(Ns)),
    ":",
    label=r"$N^{-1}$",
    c="xkcd:brick red",
)

plt.loglog(
    Ns,
    errs_var[0] * (Ns[0] / np.array(Ns)) ** 2,
    ":",
    label=r"$N^{-2}$",
    c="xkcd:navy",
)

plt.legend()

plt.xlabel(r"$\log(N)$")
plt.ylabel(r"$\log(\varepsilon)$")
if save:
    plt.savefig("figures/var_vs_fixed_z" + file_ext + ".pdf")
plt.show()

"""
Get total and relative change of the number of gates as a function of N
"""
avg_diff = []
std_div = []

for N in Ns:
    cur_diff = []
    for s in range(30):
        H = random_su_d(2, seed=s)
        trotter = Trotterization(small_pulses, N, H, coeffs="Z")
        cur_diff.append(
            np.sum(np.abs(trotter.variable_weights()[0]))
            - sum(np.abs(trotter.fixed_weights()[0])) * N
        )
    avg_diff.append(np.mean(cur_diff))
    std_div.append(np.std(cur_diff))

avg_diff = np.array(avg_diff)
std_div = np.array(std_div)

plt.plot(Ns, avg_diff)
plt.fill_between(Ns, avg_diff - std_div, avg_diff + std_div, alpha=0.2)
plt.xlabel(r"$N$")
plt.ylabel(r"$\Delta L$")
if save:
    plt.savefig("figures/total_gates_seeds.pdf")
plt.show()

"""
Get total and relative change of the number of gates for fixed target
"""
gates_fixed = []
gates_var = []

for N in Ns:
    trotter = Trotterization(small_pulses, N, M, coeffs="Z")
    gates_fixed.append(sum(np.abs(trotter.fixed_weights()[0])) * N)
    gates_var.append(sum(sum(np.abs(trotter.variable_weights()[0]))))

gates_fixed = np.array(gates_fixed)
gates_var = np.array(gates_var)

plt.plot(Ns, (gates_var - gates_fixed))
plt.xlabel(r"$N$")
plt.ylabel(r"$\Delta L$")
if save:
    plt.savefig("figures/total_gates" + file_ext + ".pdf")
plt.show()


plt.plot(Ns, (gates_var - gates_fixed) / gates_fixed)
plt.xlabel(r"$N$")
plt.ylabel(r"$\Delta L/L_0$")
if save:
    plt.savefig("figures/relative_gates" + file_ext + ".pdf")
plt.show()
