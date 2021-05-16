import numpy as np
from schroedinger import *
from plot import *

system = d1schroedinger(F = 1, L = 10)
x = np.linspace(0, system.L, 1000)
for n in range(10):
    plt.plot(x, -system.waveFun(x, n))

plt.grid()
plt.show()