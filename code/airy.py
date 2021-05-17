import numpy as np
from schroedinger import *
from plot import *

x = np.linspace(-10, 1, 1000)
ai, aip, bi, bip = special.airy(x)
plt.figure(figsize=(4,3))
plt.plot(x, ai, label="$y=\mathrm{Ai}(x)$")
plt.plot(x, bi, label="$y=\mathrm{Bi}(x)$")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.grid()
plt.savefig("../figs/airy.pdf")
plt.show()