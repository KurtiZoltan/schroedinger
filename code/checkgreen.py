import numpy as np
from schroedinger import *
from plot import *

test = d1schroedinger(L=10)

E = 5

x = np.linspace(0, test.L, 10000)
y = test.L/2
G = test.G(x,y,E)
I = E*G+np.gradient(np.gradient(G, x[1]-x[0]), x[1]-x[0])-test.F*x*G
plt.plot(x[np.abs(x-y)<0.01], I[np.abs(x-y)<0.01],"o")
plt.grid()
plt.show()

print(np.sum(I)*(x[1]-x[0])) # should be ~1