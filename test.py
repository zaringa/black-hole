from vpython import sphere, vector, rate, color, scene
import random
import math

scene.width  = 1920
scene.height = 1080
scene.fullscreen = True            # можно не задавать, подтянется под экран
from vpython import sphere, vector, rate, color
import random
import math

# --- параметры ---
N = 80             # число шариков
R = 15             # зона моделирования (радиус начальных позиций)
ball_radius = 0.2  # радиус шарика
dt = 0.01          # шаг интегрирования
G = 10.0           # гравитационная постоянная
M = 100.0         # масса чёрной дыры
R_event = 0.5      # радиус горизонта событий

# горизонт событий (чёрная дыра)
event_horizon = sphere(pos=vector(0,0,0), radius=R_event, color=color.black)

# создаём шарики
balls = []
velocities = []

for _ in range(N):
    while True:
        theta = random.uniform(0, math.pi*2)
        phi = random.uniform(0, math.pi)
        r = random.uniform(R*0.4, R*0.6)

        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)

        if r > R_event + ball_radius:  # не внутри горизонта
            break

    pos = vector(x, y, z)

    # начальная скорость (орбитальная)
    vx = -0.3 * y + 0.05 * random.uniform(-1,1)
    vy =  0.3 * x + 0.05 * random.uniform(-1,1)
    vz =  0.05 * random.uniform(-1,1)
    vel = vector(vx, vy, vz)

    ball = sphere(
        pos=pos,
        radius=ball_radius,
        color=color.orange,
        make_trail=True,
        retain=200
    )
    balls.append(ball)
    velocities.append(vel)

# --- функция для окраски в зависимости от расстояния ---
def get_color(r):
    if r > 5:       # далеко
        return color.orange
    elif r > 2:     # ближе
        return color.red
    else:           # почти у горизонта
        return color.yellow

# --- анимация ---
while True:
    rate(100)
    for i, ball in enumerate(balls):
        if ball is None:
            continue

        r_vec = ball.pos
        r = r_vec.mag

        # горизонт событий
        if r <= R_event:
            ball.visible = False
            balls[i] = None
            continue

        # сила притяжения
        force = -G * M * r_vec / (r**3 + 1e-6)
        a = force

        # обновление движения
        velocities[i] += a * dt
        ball.pos += velocities[i] * dt

        # меняем цвет в зависимости от расстояния
        ball.color = get_color(r)