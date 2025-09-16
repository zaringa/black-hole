import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- параметры ---
G = 10.0           # гравитационная постоянная (условная)
M = 1000.0         # масса чёрной дыры
N = 300           # число частиц ("материя звезды")

# начальные позиции частиц (около звезды)
radius = 10.0
angles = np.random.rand(N) * 2 * np.pi
r = radius * (0.8 + 0.4 * np.random.rand(N))
x = r * np.cos(angles)
y = r * np.sin(angles)

# скорости (слегка закрученные)
vx = -0.3 * y + 0.05 * np.random.randn(N)
vy = 0.3 * x + 0.05 * np.random.randn(N)

# шаг интегрирования
dt = 0.01

# --- настройка графика ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect("equal")

# чёрная дыра в центре
ax.plot(0, 0, "ko", markersize=12)

scat = ax.scatter(x, y, s=10, color="orange")

# --- функция обновления ---
def update(frame):
    global x, y, vx, vy

    # расстояния до центра
    r = np.sqrt(x**2 + y**2)

    # сила притяжения
    ax_force = -G * M * x / (r**3 + 1e-3)
    ay_force = -G * M * y / (r**3 + 1e-3)

    # обновляем скорости
    vx += ax_force * dt
    vy += ay_force * dt

    # обновляем позиции
    x += vx * dt
    y += vy * dt

    scat.set_offsets(np.c_[x, y])
    return scat,

ani = FuncAnimation(fig, update, frames=1000, interval=20, blit=True)
plt.show()
