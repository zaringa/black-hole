import pyvista as pv
import numpy as np

class BlackHole:
    def __init__(self, mass=1.0, position=(0, 0, 0)):
        self.mass = mass
        self.position = np.array(position)
        self.radius = 2 * mass  # условный радиус горизонта событий
        
        self.event_horizon = None
        self.accretion_disk = None
        self.jets = []
        
        self.create_components()
    
    def create_components(self):
        """Создание компонентов черной дыры"""
        self.event_horizon = pv.Sphere(radius=self.radius, center=self.position)
        
        self.accretion_disk = AccretionDisk(
            inner_radius=self.radius * 1.5,
            outer_radius=self.radius * 4.0,
            center=self.position
        )
        self.jets = [
            Jet(center=self.position, direction=(0, 0, 1), length=self.radius * 3),
            Jet(center=self.position, direction=(0, 0, -1), length=self.radius * 3)
        ]
    
    def update(self, time):
        """Обновление состояния черной дыры"""
        self.accretion_disk.rotate(time)
        
        for jet in self.jets:
            jet.pulsate(time)
    
    def add_to_plotter(self, plotter):
        """Добавление компонентов на сцену"""
        plotter.add_mesh(self.event_horizon, color='black', smooth_shading=True)
        self.accretion_disk.add_to_plotter(plotter)
        
        for jet in self.jets:
            jet.add_to_plotter(plotter)


class AccretionDisk:
    def __init__(self, inner_radius=2.0, outer_radius=5.0, center=(0, 0, 0), inclination=30):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.center = np.array(center)
        self.inclination = np.radians(inclination)  # наклон в радианах
        self.rotation_angle = 0
        
        self._create_geometry()
    
    def _create_geometry(self):
        """Создание геометрии диска"""
        theta = np.linspace(0, 2*np.pi, 100)
        r = np.linspace(self.inner_radius, self.outer_radius, 50)
        
        R, T = np.meshgrid(r, theta)
        
        # Параметрические уравнения диска с наклоном
        x = R * np.cos(T)
        y = R * np.sin(T) * np.cos(self.inclination)
        z = R * np.sin(T) * np.sin(self.inclination)
        
        self.mesh = pv.StructuredGrid(x, y, z)
        
        # Создание цветового градиента (температура)
        radial_distance = np.sqrt(x**2 + y**2 + z**2)
        self.temperature = (radial_distance - self.inner_radius) / (self.outer_radius - self.inner_radius)
    
    def rotate(self, time):
        """Вращение диска"""
        self.rotation_angle = time * 0.5
        
        theta = np.linspace(0, 2*np.pi, 100)
        r = np.linspace(self.inner_radius, self.outer_radius, 50)
        
        R, T = np.meshgrid(r, theta)
        
        # Вращение + волновая деформация
        x = R * np.cos(T + self.rotation_angle)
        y = R * np.sin(T + self.rotation_angle) * np.cos(self.inclination)
        z = R * np.sin(T + self.rotation_angle) * np.sin(self.inclination) + 0.1 * np.sin(3*T + time)
        
        new_points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        self.mesh.points = new_points
        
        # Обновление температур
        radial_distance = np.sqrt(x**2 + y**2 + z**2)
        self.temperature = (radial_distance - self.inner_radius) / (self.outer_radius - self.inner_radius)
    
    def add_to_plotter(self, plotter):
        """Добавление диска на сцену"""
        plotter.add_mesh(self.mesh, scalars=self.temperature.ravel(), 
                        cmap='plasma', opacity=0.8, name='accretion_disk')


class Jet:
    def __init__(self, center=(0, 0, 0), direction=(0, 0, 1), length=3.0):
        self.center = np.array(center)
        self.direction = np.array(direction)
        self.length = length
        self.base_radius = 0.3
        
        self._create_geometry()
    
    def _create_geometry(self):
        """Создание геометрии струи"""
        jet_center = self.center + self.direction * self.length / 2
        self.mesh = pv.Cone(center=jet_center, direction=self.direction,
                           radius=self.base_radius, height=self.length)
    
    def pulsate(self, time):
        """Пульсация струи"""
        scale = 1 + 0.3 * np.sin(time * 2)  # пульсация
        current_radius = self.base_radius * scale
        current_length = self.length * scale
        
        jet_center = self.center + self.direction * current_length / 2
        self.mesh = pv.Cone(center=jet_center, direction=self.direction,
                           radius=current_radius, height=current_length)
    
    def add_to_plotter(self, plotter):
        """Добавление струи на сцену"""
        plotter.add_mesh(self.mesh, color='cyan', opacity=0.6, name='jet')


class BlackHoleScene:
    def __init__(self):
        self.plotter = pv.Plotter()
        self.black_holes = []
        self.time = 0
        
    def add_black_hole(self, mass=1.0, position=(0, 0, 0)):
        """Добавление черной дыры в сцену"""
        black_hole = BlackHole(mass, position)
        self.black_holes.append(black_hole)
        black_hole.add_to_plotter(self.plotter)
        return black_hole
    
    def animate(self, duration=10, fps=30):
        """Запуск анимации"""
        self.plotter.open_gif("black_hole_animation.gif", fps=fps)
        
        total_frames = duration * fps
        for frame in range(total_frames):
            self.time = frame / fps
            
            for black_hole in self.black_holes:
                black_hole.update(self.time)
            
            # Вращение камеры
            camera_distance = 10
            camera_x = camera_distance * np.cos(self.time * 0.3)
            camera_y = camera_distance * np.sin(self.time * 0.3)
            self.plotter.camera_position = [
                (camera_x, camera_y, 3),
                (0, 0, 0),
                (0, 0, 1)
            ]
            
            self.plotter.write_frame()
        
        self.plotter.close()
    
    def show(self):
        """Показать статичную сцену"""
        self.plotter.camera_position = [(15, 0, 3), (0, 0, 0), (0, 0, 1)]
        self.plotter.show()


if __name__ == "__main__":
    scene = BlackHoleScene()
    
    # Добавление черной дыры
    scene.add_black_hole(mass=1.0, position=(0, 0, 0))
    
    # Запуск анимации
    scene.animate(duration=5, fps=24)
    
    #scene.show()