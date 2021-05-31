import numpy as np
from schroedinger import *
from plot import *

test = d1schroedinger(L=10)

E = 5
r = 10

x = np.linspace(0, test.L, 10000)
y = test.L/2
G = test.G(x,y,E)
I = E*G+np.gradient(np.gradient(G, x[1]-x[0]), x[1]-x[0])-test.F*x*G
plt.figure(figsize=(4,3))
plt.plot(x[np.abs(x-y)<r], I[np.abs(x-y)<r], label="$\hat{H}_xG(x,y;E)$")
plt.grid()
plt.xlabel("$ax$")
plt.legend()
plt.savefig("../figs/checkgreen.pdf")
plt.show()

print(np.sum(I)*(x[1]-x[0])) # should be ~1