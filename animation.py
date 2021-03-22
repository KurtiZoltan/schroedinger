import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from schroedinger import d1schroedinger

x0 = 5
d = 1
k = 2

animTime = 120
FPS = 30
timeSpeed = 0.8

save = False

def gauss(x):
    return np.exp(-1/2 * ((x - x0)/(d))**2 + 1j*k*x)
    #return x * (L - x)

test = d1schroedinger(gauss)

fig = plt.figure(figsize = [5, 4], dpi = 300)
ax = plt.axes(xlim=(0, test.getLength()), ylim=(0, 1))
line, = ax.plot([], [], lw=1)
plt.grid()

def animate(i):
    x = np.linspace(0, test.L, 200)
    y = np.abs(test.timeEvolution(i/FPS * timeSpeed)) ** 2
    line.set_data(x, y)
    return line,

anim = animation.FuncAnimation(fig, animate, frames=animTime*FPS, interval=1000 / FPS, blit=True)

if save:
    anim.save('pingpong.mp4', fps=FPS, extra_args=['-vcodec', 'libx264'])
else:
    plt.show()