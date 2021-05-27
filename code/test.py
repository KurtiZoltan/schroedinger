import numpy as np
from scipy import special
import matplotlib.pyplot as plt

x = np.linspace(1e2, 1e4, 1000)
ai, aip, bi, bip = special.airy(-x+0.1j)
plt.plot(x, np.real(ai/bi))
plt.plot(x, np.imag(ai/bi))
plt.grid()
plt.show()