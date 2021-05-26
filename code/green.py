import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from schroedinger import *
from plot import *

test = d1schroedinger(L=10)

E = 5
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
ys = test.L*np.array([0.2,0.4,0.6,0.8])

plt.figure(figsize=(4,3))
for y in ys:
    plt.plot(x, test.G(x, y, E), label=f"$ay={y:.1f}$")
plt.xlabel("ax")
plt.ylabel("$F/a^2G(x,y,E)$")
plt.grid()
plt.legend()
plt.savefig("../figs/1dgreens.pdf")
plt.show()

# plt.imshow(np.abs(G), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
# plt.colorbar()
# plt.show()
# plt.imshow(np.abs(Galt), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
# plt.colorbar()
# plt.show()
# plt.imshow(G-Galt, aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
# plt.colorbar()
# plt.show()
x = np.linspace(0, test.L, 500)
y = np.linspace(0, test.L, 500)
x, y = np.meshgrid(x, y)
G = test.G(x,y,E)
fig = go.Figure(data=[go.Surface(x=x, y=y, z=np.real(G))])
fig.update_layout(scene = dict(
    xaxis_title=r"ax",
    yaxis_title=r"ay",
    zaxis_title=r"G(x, y; E)*F/a^2"))
fig.update_traces(showscale=False)
fig.show()