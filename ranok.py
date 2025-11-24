import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

# --- Константы сцены ---
goal_x, goal_y = 8, 8
funnel_radius = 1000

square_x, square_y, square_size = 2, 7, 1.2

# Треугольник
tx1, ty1 = 4, 8
tx2, ty2 = 6, 8
tx3, ty3 = 5, 6

# Пятиконечная звезда
star_center_x, star_center_y, star_scale = 7, 3, 0.4

# --- Сетка (RECTANGLE + RECTBMP) ---
res = 500
x = np.linspace(0, 10, res)
y = np.linspace(0, 10, res)
X, Y = np.meshgrid(x, y)

# --- Воронка цели ---
W_goal = -(funnel_radius - (X - goal_x)**2 - (Y - goal_y)**2)

# --- Квадрат ---
mask_square = (1 - np.maximum(np.abs(X - square_x), np.abs(Y - square_y) / square_size)) > 0
W_square = np.zeros_like(X)
W_square[mask_square] = 1e6

# --- Треугольник ---
T1 = (ty1 - ty2)*X - (tx1 - tx2)*Y + (ty2*tx1 - ty1*tx2)
T2 = (ty2 - ty3)*X - (tx2 - tx3)*Y + (ty3*tx2 - ty2*tx3)
T3 = (ty3 - ty1)*X - (tx3 - tx1)*Y + (ty1*tx3 - ty3*tx1)
mask_triangle = (T1 >= 0) & (T2 >= 0) & (T3 >= 0)
W_triangle = np.zeros_like(X)
W_triangle[mask_triangle] = 1e6

# --- Пятиконечная звезда ---
sx = np.array([
    star_center_x+0*star_scale, star_center_x+2.5*star_scale, star_center_x+4.76*star_scale,
    star_center_x+3.09*star_scale, star_center_x+3.82*star_scale, star_center_x+1.55*star_scale,
    star_center_x+0*star_scale, star_center_x-1.55*star_scale, star_center_x-3.82*star_scale,
    star_center_x-3.09*star_scale, star_center_x-4.76*star_scale, star_center_x-2.5*star_scale
])
sy = np.array([
    star_center_y+5*star_scale, star_center_y+1.55*star_scale, star_center_y+1.55*star_scale,
    star_center_y-0.95*star_scale, star_center_y-4.05*star_scale, star_center_y-2.5*star_scale,
    star_center_y-4.05*star_scale, star_center_y-2.5*star_scale, star_center_y-4.05*star_scale,
    star_center_y-0.95*star_scale, star_center_y+1.55*star_scale, star_center_y+1.55*star_scale
])
star_path = Path(np.column_stack([sx, sy]))
points = np.column_stack([X.ravel(), Y.ravel()])
mask_star = star_path.contains_points(points).reshape(X.shape)
W_star = np.zeros_like(X)
W_star[mask_star] = 1e6

# --- Объединение всех полей ---
W = W_goal.copy()
W[mask_square | mask_triangle | mask_star] = 1e6

# --- Нормализация как в РАНОК2Д ---
W_clipped = np.clip(W, -1000, 1000)
W_norm = (W_clipped - W_clipped.min()) / (W_clipped.max() - W_clipped.min())

# --- Отрисовка ---
plt.figure(figsize=(6,6))
plt.imshow(W_norm, extent=(0,10,0,10), origin='lower', cmap='gray')
plt.title("Потенциальное поле (эквивалент РАНОК2Д)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.show()
