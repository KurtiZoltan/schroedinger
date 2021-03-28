import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from schroedinger import *

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "New Century Schoolbook", "Bookman", "Computer Modern Roman"],
    "font.sans-serif": ["Helvetica", "Avant Garde", "Computer Modern Sans serif"],
    "font.cursive": "Zapf Chancery",
    "font.monospace": ["Courier", "Computer Modern Typewriter"],
    "text.usetex": True,
    "lines.linewidth": 1,
    'figure.autolayout': True})

Fx = 1
Fy = 1
Lx = 10
Ly = 15

x0 = 5
y0 = 3
d = np.sqrt(2)
kx = 0
ky = 2

def gauss(x, y):
    return np.exp(-((x - x0)**2 + (y - y0)**2)/(2*d**2) + 1j*(kx*x + ky*y))

test = d2schroedinger(gauss, Fx = Fx, Fy = Fy, numPoints=1000)

dx = test.x[1] - test.x[0]
dy = test.y[1] - test.y[0]

t = np.linspace(0, 30, 300)
x = np.zeros(t.shape)
x2 = np.zeros(t.shape)
xcoords, ycoords = np.meshgrid(test.x, test.y)
xcoords2 = xcoords**2
for i, currt in zip(range(len(t)), t):
    state = test.timeEvolution(currt)
    stateabssquared = np.real(state * np.conjugate(state))
    x[i] = np.sum(xcoords * stateabssquared) * dx * dy
    x2[i] = np.sum(xcoords2 * stateabssquared) * dx * dy
    print(i)
plt.plot(t, x)
plt.plot(t, x2 - x**2)

plt.grid()
plt.show()
