import bpy
import random
import math

# Очистка сцены
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Создаём ствол
height = random.uniform(4, 7)  # высота ствола
bpy.ops.mesh.primitive_cylinder_add(
    vertices=16,
    radius=0.2,
    depth=height,
    location=(0, 0, height/2)
)
trunk = bpy.context.active_object
trunk.name = "Trunk"

# Создаём крону (сферу из нескольких шаров)
for i in range(random.randint(3, 6)):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    z = height + random.uniform(0, 2)

    scale = random.uniform(1.5, 2.5)

    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=3,
        radius=1,
        location=(x, y, z)
    )
    sphere = bpy.context.active_object
    sphere.scale = (scale, scale, scale)
    sphere.name = f"Leaf_{i}"

    # Сделаем зелёный материал
    mat = bpy.data.materials.new(name=f"LeavesMat_{i}")
    mat.diffuse_color = (0, random.uniform(0.5, 0.9), 0, 1)
    sphere.data.materials.append(mat)

# Материал для ствола
mat_trunk = bpy.data.materials.new(name="TrunkMat")
mat_trunk.diffuse_color = (0.3, 0.15, 0.05, 1)
trunk.data.materials.append(mat_trunk)

print("Случайное дерево создано!")
