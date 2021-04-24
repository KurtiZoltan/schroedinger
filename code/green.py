import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from schroedinger import *
from plot import *

test = d1schroedinger(F = 0.1)

n = 5
E = 1+2j
'''
x = np.linspace(0, test.L, 1000)
y = np.linspace(0, test.L, n)
for curry in y[1:-1]:
    G = test.G(x, curry, E)
    plt.plot(x, np.real(G))
plt.grid()
plt.show()
'''

x = np.linspace(0, test.L, 1000)
y = np.linspace(0, test.L, 1000)
x, y = np.meshgrid(x, y)
G = test.G(x, y, E)

plt.imshow(np.abs(G), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
plt.colorbar()
plt.show()
'''
fig = go.Figure(data=[go.Surface(x=x, y=y, z=np.real(G))])
fig.update_layout(scene = dict(
    xaxis_title=r"x",
    yaxis_title=r"y",
    zaxis_title=r"G(x, y; E)"))
fig.show()
'''