import numpy as np
from schroedinger import *
from plot import *

system = d1schroedinger(F = 1, L = 8)
x = np.linspace(0, system.L, 1000)
plt.figure(figsize=(4,3))
for n in range(4):
    plt.plot(x, -system.waveFun(x, n), label=f"$n={n:d}$")

plt.grid()
plt.legend()
plt.xlabel("$ax$")
plt.ylabel("$\\psi/\\sqrt{a}$")
plt.savefig("../figs/allapotok.pdf")
plt.show()