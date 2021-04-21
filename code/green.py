import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from schroedinger import *
from plot import *

test = d1schroedinger(F = 1)

n = 1000
E = 5
x = np.linspace(0, test.L, n)
y = np.linspace(0, test.L, n)
G = test.G(x, x, E)
plt.plot(x, G)
plt.show()
x, y = np.meshgrid(x, y)
G = test.G(x, y, E)

plt.imshow(G, aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, G)
plt.show()