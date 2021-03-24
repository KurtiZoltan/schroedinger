import matplotlib.pyplot as plt
import numpy as np
from schroedinger import *

def semiclassical(E):
    F = test.F
    L = test.L
    n1 = 1/np.pi * 2 / 3 / F * (np.power(E, 3/2))
    n2 = 1/np.pi * 2 / 3 / F * (np.power(E, 3/2) - np.power(np.maximum(E - F*L, 0), 3/2))
    n1mask = E < L*F
    return n1 * n1mask + n2 * (1-n1mask)

test = d1schroedinger(F = 1)
test.eLevel(75)
elevels = test.Es
x = np.linspace(0, elevels[-1], 10000)
y = np.zeros(x.shape)
for i, currentx in zip(range(len(x)), x):
    y[i] = np.sum(elevels < currentx)

plt.plot(x, y)
plt.plot(x, semiclassical(x))
plt.grid()
plt.show()