import numpy as np
from matplotlib import animation
from schroedinger import *
from plot import *

x0 = 5
d = 1
k = 2

animTime = 120
FPS = 30
timeSpeed = 0.5

save = False

def gauss(x):
    return np.exp(-1/2 * ((x - x0)/(d))**2 + 1j*k*x)
    #return x * (L - x)

test = d1schroedinger(gauss, F = 0.1)

fig = plt.figure(figsize = [16, 10], dpi = 108)
plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title
ax = plt.axes(xlim=(0, test.getLength()), ylim=(0, 1))
line, = ax.plot([], [], lw=3)
plt.xlabel("$ax$")
plt.ylabel("$|\psi(x)|^2/a$")
plt.grid()

def animate(i):
    x = test.x
    y = np.abs(test.timeEvolution(i/FPS * timeSpeed)) ** 2
    line.set_data(x, y)
    return line,

anim = animation.FuncAnimation(fig, animate, frames=animTime*FPS, interval=1000 / FPS, blit=True)

if save:
    anim.save("../videos/" + "final1d.mp4", fps=FPS, extra_args=['-vcodec', 'libx264'])
else:
    plt.show()
