import numpy as np
import pyvista as pv
import warnings
import time

from black_hole_lib import (
    isco_radius,
    normalized_disk_temperature,
    orbital_angular_velocity,
    radial_inflow_velocity,
    schwarzschild_radius,
)


def _sanitize_points(points, fallback):
    points = np.asarray(points, dtype=float)
    fallback = np.asarray(fallback, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    bad = ~np.isfinite(points).all(axis=1)
    if np.any(bad):
        points = points.copy()
        points[bad] = fallback[None, :]
    return points


class BlackHole:
    def __init__(self, mass=1.0, position=(0, 0, 0)):
        self.mass = float(mass)
        self.position = np.asarray(position, dtype=float)
        self.radius = schwarzschild_radius(self.mass, g_const=1.0, c_const=1.0)

        self.uid = f"bh_{id(self)}"
        self.event_horizon = None
        self.accretion_disk = None
        self.particles = None
        self.planet = None
        self.jets = []

        self.create_components()

    def create_components(self):
        """Создание компонентов черной дыры."""
        self.event_horizon = pv.Sphere(
            radius=self.radius,
            center=self.position,
            theta_resolution=64,
            phi_resolution=64,
        )

        disk_inner = max(isco_radius(self.mass, g_const=1.0, c_const=1.0), self.radius * 1.15)
        disk_outer = disk_inner * 2.8

        self.accretion_disk = AccretionDisk(
            mass=self.mass,
            inner_radius=disk_inner,
            outer_radius=disk_outer,
            center=self.position,
            inclination=28,
        )
        self.particles = OrbitalParticles(
            mass=self.mass,
            inner_radius=disk_inner * 1.02,
            outer_radius=disk_outer * 0.98,
            center=self.position,
            inclination=28,
            count=1200,
            name=f"{self.uid}_particles",
        )
        self.planet = Planet(
            host_mass=self.mass,
            bh_radius=self.radius,
            center=self.position,
            orbital_radius=disk_inner * 1.08,
            inclination=28,
            radius=self.radius * 0.42,
            name=f"{self.uid}_planet",
        )

        jet_length = self.radius * 4.0
        self.jets = [
            Jet(
                center=self.position,
                direction=(0, 0, 1),
                length=jet_length,
                phase_offset=0.0,
                name=f"{self.uid}_jet_north",
            ),
            Jet(
                center=self.position,
                direction=(0, 0, -1),
                length=jet_length,
                phase_offset=np.pi,
                name=f"{self.uid}_jet_south",
            ),
        ]

    def update(self, dt, time):
        """Обновление состояния черной дыры."""
        accretion_level = self.accretion_disk.evolve(dt, time)
        self.particles.evolve(dt, time, accretion_level=accretion_level)
        self.planet.evolve(dt, time)
        jet_power = np.clip(0.8 + 2.5 * accretion_level, 0.65, 1.7)

        for jet in self.jets:
            jet.pulsate(time, power=jet_power)

    def add_to_plotter(self, plotter):
        """Добавление компонентов на сцену."""
        plotter.add_mesh(
            self.event_horizon,
            color="#0e0e10",
            smooth_shading=True,
            metallic=0.2,
            roughness=0.6,
            name=f"{self.uid}_event_horizon",
        )
        self.accretion_disk.add_to_plotter(plotter, name=f"{self.uid}_accretion_disk")
        self.particles.add_to_plotter(plotter)
        self.planet.add_to_plotter(plotter)

        for jet in self.jets:
            jet.add_to_plotter(plotter)


class AccretionDisk:
    def __init__(
        self,
        mass=1.0,
        inner_radius=2.0,
        outer_radius=5.0,
        center=(0, 0, 0),
        inclination=30,
        radial_resolution=90,
        angular_resolution=180,
    ):
        self.mass = float(mass)
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.center = np.asarray(center, dtype=float)
        self.inclination = np.radians(inclination)

        self.radial_resolution = int(radial_resolution)
        self.angular_resolution = int(angular_resolution)

        self.rng = np.random.default_rng(42)
        self.r_s = schwarzschild_radius(self.mass, g_const=1.0, c_const=1.0)
        self.mesh = None
        self._points_view = None
        self.temperature = None

        self.radius_grid = None
        self.phase_grid = None
        self.accretion_level = 0.0

        self._create_geometry()

    def _create_geometry(self):
        radii = np.linspace(self.inner_radius, self.outer_radius, self.radial_resolution)
        phases = np.linspace(0.0, 2.0 * np.pi, self.angular_resolution, endpoint=False)
        self.radius_grid, self.phase_grid = np.meshgrid(radii, phases)
        self._update_mesh(time=0.0)

    def _update_mesh(self, time):
        local_radius = np.maximum(self.radius_grid, self.inner_radius + 1e-5)

        x = local_radius * np.cos(self.phase_grid)
        y_plane = local_radius * np.sin(self.phase_grid)

        # Мягкая вертикальная деформация, уменьшающаяся с радиусом.
        warp = (
            0.07
            * self.inner_radius
            * np.sin(2.0 * self.phase_grid + 0.8 * time)
            * (self.inner_radius / local_radius) ** 0.5
        )

        y = y_plane * np.cos(self.inclination)
        z = y_plane * np.sin(self.inclination) + warp

        x += self.center[0]
        y += self.center[1]
        z += self.center[2]
        new_points = _sanitize_points(
            np.column_stack((x.ravel(order="C"), y.ravel(order="C"), z.ravel(order="C"))),
            fallback=self.center,
        )

        if self.mesh is None:
            self.mesh = pv.StructuredGrid(x, y, z)
            self._points_view = self.mesh.points
        else:
            if self._points_view is None or self._points_view.shape != new_points.shape:
                self._points_view = self.mesh.points
            self._points_view[:] = new_points

        self._update_temperature()

    def _update_temperature(self):
        base = normalized_disk_temperature(self.radius_grid, self.inner_radius)
        doppler_boost = 0.14 * np.cos(self.phase_grid)
        self.temperature = np.clip(base + doppler_boost, 0.0, 1.0)
        self.mesh.point_data["temperature"] = self.temperature.ravel(order="C")

    def evolve(self, dt, time):
        """Физически осмысленная эволюция диска за шаг времени."""
        omega = orbital_angular_velocity(self.mass, self.radius_grid, r_s=self.r_s, g_const=1.0)
        radial_speed = radial_inflow_velocity(
            self.mass,
            self.radius_grid,
            r_s=self.r_s,
            alpha=0.045,
            g_const=1.0,
        )

        self.phase_grid = np.mod(self.phase_grid + omega * dt, 2.0 * np.pi)
        self.radius_grid = self.radius_grid + radial_speed * dt

        consumed_mask = self.radius_grid <= self.inner_radius * 1.02
        if np.any(consumed_mask):
            reset_count = int(np.count_nonzero(consumed_mask))
            self.radius_grid[consumed_mask] = self.rng.uniform(
                self.outer_radius * 0.9,
                self.outer_radius,
                size=reset_count,
            )
            self.phase_grid[consumed_mask] = self.rng.uniform(0.0, 2.0 * np.pi, size=reset_count)

        self._update_mesh(time)

        inner_band = self.radius_grid <= self.inner_radius * 1.2
        if np.any(inner_band):
            flow = float(np.mean(-radial_speed[inner_band]))
            v_orb = float(np.mean(omega[inner_band] * self.radius_grid[inner_band]))
        else:
            flow = float(np.mean(-radial_speed))
            v_orb = float(np.mean(omega * self.radius_grid))

        self.accretion_level = float(np.clip(flow / (v_orb + 1e-6), 0.0, 1.0))
        return self.accretion_level

    def add_to_plotter(self, plotter, name="accretion_disk"):
        """Добавление диска на сцену."""
        plotter.add_mesh(
            self.mesh,
            scalars="temperature",
            cmap="inferno",
            clim=[0.0, 1.0],
            smooth_shading=True,
            opacity=0.9,
            name=name,
        )


class OrbitalParticles:
    def __init__(
        self,
        mass=1.0,
        inner_radius=2.0,
        outer_radius=5.0,
        center=(0, 0, 0),
        inclination=30,
        count=800,
        name="orbital_particles",
    ):
        self.mass = float(mass)
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.center = np.asarray(center, dtype=float)
        self.inclination = np.radians(inclination)
        self.count = int(count)
        self.name = name

        self.rng = np.random.default_rng(123)
        self.r_s = schwarzschild_radius(self.mass, g_const=1.0, c_const=1.0)

        self.radius = None
        self.phase = None
        self.height = None
        self.flicker_phase = None
        self.state = None  # 0: disk, 1: plunge, 2: center-fall
        self.pr = None
        self.energy = None
        self.lz = None
        self.mesh = None
        self.trail_mesh = None
        self.prev_points = None
        self.glow = None

        self._seed_particles()
        self._update_mesh(time=0.0)

    def _seed_particles(self):
        self.radius = np.zeros(self.count, dtype=float)
        self.phase = np.zeros(self.count, dtype=float)
        self.height = np.zeros(self.count, dtype=float)
        self.flicker_phase = np.zeros(self.count, dtype=float)
        self.state = np.zeros(self.count, dtype=np.int8)
        self.pr = np.zeros(self.count, dtype=float)
        self.energy = np.zeros(self.count, dtype=float)
        self.lz = np.zeros(self.count, dtype=float)
        self._respawn(np.arange(self.count))

    def _respawn(self, idx):
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            return

        spawn_min = max(self.inner_radius * 1.04, self.r_s * 1.2)
        spawn_max = max(self.outer_radius, spawn_min + 1e-3)
        self.radius[idx] = self.rng.uniform(spawn_min, spawn_max, size=idx.size)
        self.phase[idx] = self.rng.uniform(0.0, 2.0 * np.pi, size=idx.size)
        self.height[idx] = self.rng.normal(0.0, 0.035 * self.inner_radius, size=idx.size)
        self.flicker_phase[idx] = self.rng.uniform(0.0, 2.0 * np.pi, size=idx.size)
        self.state[idx] = 0
        self.pr[idx] = 0.0
        self.energy[idx] = 0.0
        self.lz[idx] = 0.0

    def _start_plunge(self, idx, radial_speed, omega):
        if idx.size == 0:
            return

        r0 = np.maximum(self.radius[idx], self.r_s * 1.005)
        lz = np.maximum(r0 ** 2 * omega[idx], 1e-6)
        pr0 = np.minimum(radial_speed[idx], -1e-6)

        f0 = np.clip(1.0 - self.r_s / r0, 1e-9, None)
        # (dr/dtau)^2 = E^2 - (1-r_s/r)*(1 + L^2/r^2), c=1 in sim units.
        e2 = pr0 ** 2 + f0 * (1.0 + (lz ** 2) / np.maximum(r0 ** 2, 1e-9))

        self.lz[idx] = lz
        self.pr[idx] = pr0
        self.energy[idx] = np.sqrt(np.maximum(e2, 1e-9))
        self.state[idx] = 1

    def _points(self, time):
        radius = np.clip(self.radius, 0.0, None)
        x = radius * np.cos(self.phase)
        y_plane = radius * np.sin(self.phase)

        disk_fraction = np.clip(radius / (self.inner_radius + 1e-9), 0.0, 1.0)
        local_height = (
            self.height * (0.12 + 0.88 * disk_fraction ** 1.5)
            + 0.04 * self.inner_radius * np.sin(2.5 * self.phase + 1.6 * time) * disk_fraction
        )
        y = y_plane * np.cos(self.inclination) - local_height * np.sin(self.inclination)
        z = y_plane * np.sin(self.inclination) + local_height * np.cos(self.inclination)

        return np.column_stack(
            (
                x + self.center[0],
                y + self.center[1],
                z + self.center[2],
            )
        )

    def _update_glow(self, time):
        radial_norm = 1.0 - np.clip(
            (self.radius - self.r_s) / (self.outer_radius - self.r_s + 1e-9),
            0.0,
            1.0,
        )
        flicker = 0.68 + 0.32 * np.sin(6.0 * time + self.flicker_phase)
        plunge_boost = np.where(self.state == 1, 0.14, 0.0) + np.where(self.state == 2, 0.26, 0.0)
        self.glow = np.clip(0.32 + 0.46 * radial_norm + 0.25 * flicker + plunge_boost, 0.0, 1.0)
        self.mesh.point_data["glow"] = self.glow

    def _update_trails(self, points):
        if self.prev_points is None:
            self.prev_points = points.copy()

        plunge_idx = np.where((self.state == 1) | (self.state == 2))[0]
        if plunge_idx.size == 0:
            if self.trail_mesh is None:
                self.trail_mesh = pv.PolyData(np.repeat(self.center[None, :], 1, axis=0))
            else:
                self.trail_mesh.points = np.repeat(self.center[None, :], 1, axis=0)
            self.trail_mesh.point_data["trail_glow"] = np.array([0.0], dtype=float)
            return

        curr = points[plunge_idx]
        prev = self.prev_points[plunge_idx]
        delta = curr - prev
        speed = np.linalg.norm(delta, axis=1)
        valid_motion = speed > 1e-8

        if np.any(~valid_motion):
            inward = self.center[None, :] - curr[~valid_motion]
            inward /= np.linalg.norm(inward, axis=1, keepdims=True) + 1e-9
            delta[~valid_motion] = -0.06 * inward

        seg_u = np.array([0.18, 0.42, 0.66, 0.90], dtype=float)
        seg_n = seg_u.size
        trail_points = np.repeat(curr, seg_n, axis=0)
        repeated_delta = np.repeat(delta, seg_n, axis=0)
        repeated_u = np.tile(seg_u, plunge_idx.size)
        trail_points -= repeated_delta * repeated_u[:, None]
        trail_points = _sanitize_points(trail_points, fallback=self.center)

        phase_boost = np.where(self.state[plunge_idx] == 2, 0.18, 0.0)
        base_glow = np.clip(self.glow[plunge_idx] + phase_boost, 0.0, 1.0)
        trail_glow = np.repeat(base_glow, seg_n) * np.tile(1.0 - seg_u * 0.92, plunge_idx.size)

        if self.trail_mesh is None:
            self.trail_mesh = pv.PolyData(trail_points)
        else:
            self.trail_mesh.points = trail_points
        self.trail_mesh.point_data["trail_glow"] = np.clip(trail_glow, 0.0, 1.0)

    def _update_mesh(self, time):
        points = _sanitize_points(self._points(time), fallback=self.center)
        if self.mesh is None:
            self.mesh = pv.PolyData(points.copy())
        else:
            self.mesh.points = points
        self._update_glow(time)
        self._update_trails(points)
        self.prev_points = points.copy()

    def evolve(self, dt, time, accretion_level=0.0):
        disk_mask = self.state == 0
        omega = np.zeros(self.count, dtype=float)
        radial_speed = np.zeros(self.count, dtype=float)

        if np.any(disk_mask):
            disk_r = np.maximum(self.radius[disk_mask], self.inner_radius * 1.002)
            omega[disk_mask] = orbital_angular_velocity(self.mass, disk_r, r_s=self.r_s, g_const=1.0)
            speed_boost = 1.0 + 0.35 * float(accretion_level)
            orbital_dt = dt * (1.8 + 1.3 * float(accretion_level))
            self.phase[disk_mask] = np.mod(self.phase[disk_mask] + omega[disk_mask] * orbital_dt * speed_boost, 2.0 * np.pi)

            alpha = 0.018 + 0.03 * float(accretion_level)
            radial_speed[disk_mask] = radial_inflow_velocity(
                self.mass,
                disk_r,
                r_s=self.r_s,
                alpha=alpha,
                g_const=1.0,
            )
            # Формула скорости сохраняется; ускоряем только симуляционное время, чтобы падение было видно в кадре.
            inflow_dt = dt * (14.0 + 18.0 * float(accretion_level))
            self.radius[disk_mask] = self.radius[disk_mask] + radial_speed[disk_mask] * inflow_dt

            to_plunge = disk_mask & (self.radius <= self.inner_radius * 1.01)
            if np.any(to_plunge):
                self._start_plunge(np.where(to_plunge)[0], radial_speed, omega)

        plunge_mask = self.state == 1
        if np.any(plunge_mask):
            r = np.maximum(self.radius[plunge_mask], self.r_s * 1.001)
            lz = self.lz[plunge_mask]
            energy = self.energy[plunge_mask]

            f = np.clip(1.0 - self.r_s / r, 1e-9, None)
            # Plunge phase from conserved E, Lz:
            # (dr/dtau)^2 = E^2 - (1-r_s/r)*(1 + Lz^2/r^2), dphi/dtau = Lz/r^2.
            radial_term = energy ** 2 - f * (1.0 + (lz ** 2) / np.maximum(r ** 2, 1e-9))
            pr = -np.sqrt(np.maximum(radial_term, 1e-10))
            horizon_pull = 0.05 * np.clip(self.r_s / np.maximum(r - self.r_s + 1e-6, 1e-6), 0.0, 8.0)
            pr -= horizon_pull

            dphi = lz / np.maximum(r ** 2, 1e-9)
            plunge_dt = dt * 2.4
            self.phase[plunge_mask] = np.mod(self.phase[plunge_mask] + dphi * plunge_dt, 2.0 * np.pi)
            self.radius[plunge_mask] = np.maximum(0.0, r + pr * plunge_dt)
            self.pr[plunge_mask] = pr

            to_center = plunge_mask & (self.radius <= self.r_s * 1.003)
            if np.any(to_center):
                self.state[to_center] = 2

        center_mask = self.state == 2
        if np.any(center_mask):
            r = np.maximum(self.radius[center_mask], 0.0)
            collapse_speed = (0.35 + 0.95 * np.clip(self.r_s / (r + self.r_s * 0.05 + 1e-9), 0.0, 6.0)) * self.r_s
            center_dt = dt * 2.0
            self.radius[center_mask] = np.maximum(0.0, r - collapse_speed * center_dt)
            spin = 1.2 + 2.2 * np.clip(r / (self.inner_radius + 1e-9), 0.0, 1.0)
            self.phase[center_mask] = np.mod(self.phase[center_mask] + spin * center_dt, 2.0 * np.pi)
            self.height[center_mask] *= np.maximum(0.0, 1.0 - 3.0 * center_dt)

        swallowed = self.radius <= self.r_s * 0.04
        if np.any(swallowed):
            self._respawn(np.where(swallowed)[0])

        self.radius = np.clip(np.nan_to_num(self.radius, nan=self.outer_radius, posinf=self.outer_radius, neginf=0.0), 0.0, self.outer_radius * 1.2)
        self.phase = np.mod(np.nan_to_num(self.phase, nan=0.0, posinf=0.0, neginf=0.0), 2.0 * np.pi)
        self.height = np.nan_to_num(self.height, nan=0.0, posinf=0.0, neginf=0.0)

        self._update_mesh(time)

    def add_to_plotter(self, plotter):
        """Добавление орбитальных частиц на сцену."""
        if self.trail_mesh is not None:
            plotter.add_mesh(
                self.trail_mesh,
                scalars="trail_glow",
                cmap="afmhot",
                clim=[0.0, 1.0],
                style="points",
                point_size=5.5,
                render_points_as_spheres=True,
                opacity=0.74,
                name=f"{self.name}_trails",
            )
        plotter.add_mesh(
            self.mesh,
            scalars="glow",
            cmap="afmhot",
            clim=[0.0, 1.0],
            style="points",
            point_size=7.0,
            render_points_as_spheres=True,
            opacity=0.95,
            name=self.name,
        )


class DebrisCloud:
    def __init__(self, host_mass=1.0, bh_radius=2.0, center=(0, 0, 0), max_particles=1600, name="planet_debris"):
        self.host_mass = float(host_mass)
        self.bh_radius = float(bh_radius)
        self.center = np.asarray(center, dtype=float)
        self.max_particles = int(max_particles)
        self.name = name

        self.rng = np.random.default_rng(404)
        self.points = np.repeat(self.center[None, :], self.max_particles, axis=0)
        self.velocity = np.zeros((self.max_particles, 3), dtype=float)
        self.ttl = np.zeros(self.max_particles, dtype=float)
        self.glow = np.zeros(self.max_particles, dtype=float)
        self.active = np.zeros(self.max_particles, dtype=bool)

        self.mesh = pv.PolyData(self.points.copy())
        self.mesh.point_data["glow"] = self.glow

    def emit(self, origin, tangent, count, strength=0.0, spread=0.2):
        count = int(max(0, count))
        if count == 0:
            return

        free_idx = np.where(~self.active)[0]
        if free_idx.size == 0:
            return

        use_n = min(count, free_idx.size)
        idx = free_idx[:use_n]

        origin = np.asarray(origin, dtype=float)
        tangent = np.asarray(tangent, dtype=float)
        tangent = tangent / (np.linalg.norm(tangent) + 1e-9)

        inward = self.center - origin
        inward = inward / (np.linalg.norm(inward) + 1e-9)

        rnd = self.rng.normal(size=(use_n, 3))
        rnd /= np.linalg.norm(rnd, axis=1, keepdims=True) + 1e-9

        speed = (0.45 + 1.6 * float(strength)) * self.rng.uniform(0.7, 1.3, size=use_n)
        tangential_part = tangent[None, :] * speed[:, None]
        inward_part = inward[None, :] * (0.25 + 1.0 * float(strength))
        random_part = rnd * (0.18 + 0.45 * float(strength))

        self.points[idx] = origin[None, :] + rnd * float(spread) * self.rng.uniform(0.1, 1.0, size=(use_n, 1))
        self.velocity[idx] = tangential_part + inward_part + random_part
        self.ttl[idx] = self.rng.uniform(1.0, 2.4, size=use_n) + 1.8 * float(strength)
        self.glow[idx] = self.rng.uniform(0.35, 1.0, size=use_n)
        self.active[idx] = True
        self.points[idx] = _sanitize_points(self.points[idx], fallback=self.center)
        self.velocity[idx] = np.nan_to_num(self.velocity[idx], nan=0.0, posinf=0.0, neginf=0.0)

    def evolve(self, dt):
        if not np.any(self.active):
            return

        idx = np.where(self.active)[0]
        points = self.points[idx]
        velocity = self.velocity[idx]
        ttl = self.ttl[idx]
        glow = self.glow[idx]

        rel = self.center[None, :] - points
        dist = np.linalg.norm(rel, axis=1) + 1e-9
        grav = self.host_mass * rel / (dist[:, None] ** 3)

        velocity += (grav - 0.12 * velocity) * dt
        points += velocity * dt

        ttl -= dt * (0.85 + 0.75 * np.clip(self.bh_radius / dist, 0.0, 3.0))
        glow = np.clip(glow * (1.0 - 0.25 * dt) + 0.18 * np.clip(self.bh_radius / dist, 0.0, 2.5), 0.0, 1.0)

        swallowed = dist <= self.bh_radius * 1.01
        expired = ttl <= 0.0
        dead = swallowed | expired

        self.points[idx] = points
        self.velocity[idx] = velocity
        self.ttl[idx] = ttl
        self.glow[idx] = glow

        if np.any(dead):
            dead_idx = idx[dead]
            self.active[dead_idx] = False
            self.ttl[dead_idx] = 0.0
            self.glow[dead_idx] = 0.0
            self.velocity[dead_idx] = 0.0
            self.points[dead_idx] = self.center[None, :]

        self.points = _sanitize_points(self.points, fallback=self.center)
        self.velocity = np.nan_to_num(self.velocity, nan=0.0, posinf=0.0, neginf=0.0)
        self.ttl = np.nan_to_num(self.ttl, nan=0.0, posinf=0.0, neginf=0.0)
        self.glow = np.clip(np.nan_to_num(self.glow, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        self.mesh.points = self.points
        self.mesh.point_data["glow"] = np.where(self.active, self.glow, 0.0)

    def add_to_plotter(self, plotter):
        plotter.add_mesh(
            self.mesh,
            scalars="glow",
            cmap="afmhot",
            clim=[0.0, 1.0],
            style="points",
            point_size=6.0,
            render_points_as_spheres=True,
            opacity=0.9,
            name=self.name,
        )


class Planet:
    def __init__(
        self,
        host_mass=1.0,
        bh_radius=2.0,
        center=(0, 0, 0),
        orbital_radius=8.0,
        inclination=25,
        radius=0.7,
        name="planet",
    ):
        self.host_mass = float(host_mass)
        self.bh_radius = float(bh_radius)
        self.center = np.asarray(center, dtype=float)
        self.orbital_radius = float(orbital_radius)
        self.inclination = np.radians(inclination)
        self.base_radius = float(radius)
        self.name = name

        self.rng = np.random.default_rng(2025)
        self.phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.destruction_progress = 0.0
        self.absorbed = False
        self.disruption_radius = max(self.bh_radius * 3.35, self.orbital_radius * 0.86)
        self.absorb_radius = self.bh_radius * 1.015
        self.current_radius = self.base_radius

        self.position = self._position_from_orbit()

        self.base_mesh = pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60)
        self.mesh = self.base_mesh.copy(deep=True)
        self.surface_phase = self.rng.uniform(0.0, 2.0 * np.pi, size=self.mesh.n_points)

        self.debris = DebrisCloud(
            host_mass=self.host_mass,
            bh_radius=self.bh_radius,
            center=self.center,
            max_particles=1800,
            name=f"{self.name}_debris",
        )

        self._update_mesh(time=0.0)

    def _position_from_orbit(self):
        x = self.orbital_radius * np.cos(self.phase)
        y_plane = self.orbital_radius * np.sin(self.phase)

        y = y_plane * np.cos(self.inclination)
        z = y_plane * np.sin(self.inclination)
        return self.center + np.array([x, y, z], dtype=float)

    def _tangent_direction(self):
        tangent = np.array(
            [
                -np.sin(self.phase),
                np.cos(self.phase) * np.cos(self.inclination),
                np.cos(self.phase) * np.sin(self.inclination),
            ],
            dtype=float,
        )
        return tangent / (np.linalg.norm(tangent) + 1e-9)

    def _update_mesh(self, time):
        self.current_radius = self.base_radius * (1.0 - 0.96 * self.destruction_progress)
        self.current_radius = max(self.current_radius, self.base_radius * 0.03)
        jitter = 1.0 + self.destruction_progress * (
            0.30 * np.sin(3.0 * self.surface_phase + 4.3 * time)
            + 0.18 * np.sin(7.0 * self.surface_phase + 1.7 * time)
        )
        jitter = np.clip(jitter, 0.5, 1.65)

        new_points = self.base_mesh.points * (self.current_radius * jitter[:, None]) + self.position
        self.mesh.points = new_points

        stress = np.full(self.mesh.n_points, self.destruction_progress, dtype=float)
        self.mesh.point_data["stress"] = stress

    def evolve(self, dt, time):
        if not self.absorbed:
            omega = float(
                orbital_angular_velocity(
                    self.host_mass,
                    self.orbital_radius,
                    r_s=self.bh_radius,
                    g_const=1.0,
                )
            )
            self.phase = np.mod(self.phase + omega * dt * (1.0 + 0.25 * self.destruction_progress), 2.0 * np.pi)

            horizon_ratio = np.clip(self.bh_radius / max(self.orbital_radius, self.bh_radius * 1.001), 0.0, 0.99)
            inward_rate = 0.26 + 2.4 * horizon_ratio ** 2 + 1.9 * self.destruction_progress
            self.orbital_radius = max(self.bh_radius * 1.005, self.orbital_radius - inward_rate * dt)
            self.position = self._position_from_orbit()

            if self.orbital_radius < self.disruption_radius:
                target = np.clip(
                    (self.disruption_radius - self.orbital_radius)
                    / (self.disruption_radius - self.bh_radius * 1.02 + 1e-9),
                    0.0,
                    1.0,
                )
                self.destruction_progress = max(self.destruction_progress, float(target))

            if self.destruction_progress > 0.12:
                burst = max(0.0, np.sin(16.0 * time + 3.0 * self.phase))
                emit_count = int((16.0 + 220.0 * self.destruction_progress + 140.0 * burst * self.destruction_progress) * dt)
                self.debris.emit(
                    origin=self.position,
                    tangent=self._tangent_direction(),
                    count=emit_count,
                    strength=self.destruction_progress,
                    spread=self.base_radius * (0.35 + 1.2 * self.destruction_progress),
                )

            self._update_mesh(time)

            if self.orbital_radius <= self.absorb_radius:
                self.absorbed = True
                self.destruction_progress = 1.0
                self.position = self.center.copy()
                self._update_mesh(time)

        self.debris.evolve(dt)

    def add_to_plotter(self, plotter):
        plotter.add_mesh(
            self.mesh,
            scalars="stress",
            cmap="copper",
            clim=[0.0, 1.0],
            smooth_shading=True,
            specular=0.25,
            opacity=0.95,
            name=self.name,
        )
        self.debris.add_to_plotter(plotter)


class SolarEruptionParticles:
    def __init__(
        self,
        center=(0, 0, 0),
        radius=3.0,
        max_particles=3000,
        surface_temp=5600.0,
        core_temp=15000.0,
        name="solar_eruptions",
    ):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.max_particles = int(max_particles)
        self.surface_temp = float(surface_temp)
        self.core_temp = float(core_temp)
        self.name = name

        self.rng = np.random.default_rng(888)
        self.points = np.repeat(self.center[None, :], self.max_particles, axis=0)
        self.velocity = np.zeros((self.max_particles, 3), dtype=float)
        self.ttl = np.zeros(self.max_particles, dtype=float)
        self.temperature = np.full(self.max_particles, self.surface_temp, dtype=float)
        self.active = np.zeros(self.max_particles, dtype=bool)

        self.mesh = pv.PolyData(self.points.copy())
        self.mesh.point_data["rgb"] = np.zeros((self.max_particles, 3), dtype=np.uint8)

    def _temperature_to_rgb(self, temp_norm):
        t = np.clip(np.asarray(temp_norm, dtype=float), 0.0, 1.0)
        rgb = np.zeros((t.size, 3), dtype=float)

        c0 = np.array([255.0, 110.0, 35.0])  # cooler orange
        c1 = np.array([255.0, 205.0, 115.0])  # warm yellow-orange
        c2 = np.array([255.0, 245.0, 220.0])  # near-white
        c3 = np.array([170.0, 205.0, 255.0])  # hot blue-white

        m0 = t < 0.45
        m1 = (t >= 0.45) & (t < 0.85)
        m2 = ~(m0 | m1)

        if np.any(m0):
            u = t[m0] / 0.45
            rgb[m0] = (1.0 - u)[:, None] * c0 + u[:, None] * c1
        if np.any(m1):
            u = (t[m1] - 0.45) / 0.40
            rgb[m1] = (1.0 - u)[:, None] * c1 + u[:, None] * c2
        if np.any(m2):
            u = (t[m2] - 0.85) / 0.15
            rgb[m2] = (1.0 - u)[:, None] * c2 + u[:, None] * c3

        return np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    def _refresh_colors(self):
        temp_norm = (self.temperature - self.surface_temp) / (self.core_temp - self.surface_temp + 1e-9)
        rgb = np.zeros((self.max_particles, 3), dtype=np.uint8)
        idx = np.where(self.active)[0]
        if idx.size:
            rgb[idx] = self._temperature_to_rgb(temp_norm[idx])
        self.mesh.point_data["rgb"] = rgb

    def emit(self, origin, normal, tangent, count, depth_hotness=0.5, eruption_power=0.5, emitter_velocity=None):
        count = int(max(0, count))
        if count == 0:
            return

        free_idx = np.where(~self.active)[0]
        if free_idx.size == 0:
            return

        use_n = min(count, free_idx.size)
        idx = free_idx[:use_n]

        origin = np.asarray(origin, dtype=float)
        normal = np.asarray(normal, dtype=float)
        tangent = np.asarray(tangent, dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
        side = np.cross(normal, tangent)
        side = side / (np.linalg.norm(side) + 1e-9)
        if emitter_velocity is None:
            emitter_velocity = np.zeros(3, dtype=float)
        emitter_velocity = np.asarray(emitter_velocity, dtype=float)

        rnd = self.rng.normal(size=(use_n, 3))
        rnd /= np.linalg.norm(rnd, axis=1, keepdims=True) + 1e-9

        speed = self.rng.uniform(0.9, 1.7, size=use_n) * (0.8 + 1.4 * float(eruption_power))
        self.points[idx] = origin[None, :] + normal[None, :] * (0.04 * self.radius) + rnd * (0.06 * self.radius)
        self.velocity[idx] = (
            normal[None, :] * speed[:, None]
            + tangent[None, :] * (0.35 + 0.9 * float(eruption_power))
            + side[None, :] * self.rng.normal(0.0, 0.32, size=(use_n, 1))
            + rnd * (0.2 + 0.45 * float(eruption_power))
            + emitter_velocity[None, :]
        )

        depth_hotness = float(np.clip(depth_hotness, 0.0, 1.0))
        base_temp = self.surface_temp + depth_hotness * (self.core_temp - self.surface_temp)
        self.temperature[idx] = base_temp * self.rng.uniform(0.90, 1.12, size=use_n)
        self.ttl[idx] = self.rng.uniform(1.0, 2.6, size=use_n) * (0.9 + 0.8 * float(eruption_power))
        self.active[idx] = True

        self.mesh.points = self.points
        self._refresh_colors()

    def translate(self, delta):
        delta = np.asarray(delta, dtype=float)
        self.points += delta[None, :]
        self.mesh.points = self.points

    def evolve(self, dt):
        if not np.any(self.active):
            return

        idx = np.where(self.active)[0]
        points = self.points[idx]
        velocity = self.velocity[idx]
        ttl = self.ttl[idx]
        temperature = self.temperature[idx]

        rel = self.center[None, :] - points
        dist = np.linalg.norm(rel, axis=1) + 1e-9
        gravity = 0.7 * self.radius * rel / (dist[:, None] ** 3)

        velocity += (gravity - 0.08 * velocity) * dt
        points += velocity * dt

        cooling = (1800.0 + 1500.0 * np.clip(dist / self.radius - 1.0, 0.0, 3.0)) * dt
        temperature -= cooling
        ttl -= dt * (0.9 + 0.4 * np.clip(dist / self.radius, 0.0, 4.0))

        swallowed = dist <= self.radius * 0.985
        too_cold = temperature <= self.surface_temp * 0.72
        dead = swallowed | (ttl <= 0.0) | too_cold

        self.points[idx] = points
        self.velocity[idx] = velocity
        self.ttl[idx] = ttl
        self.temperature[idx] = temperature

        if np.any(dead):
            dead_idx = idx[dead]
            self.active[dead_idx] = False
            self.points[dead_idx] = self.center[None, :]
            self.velocity[dead_idx] = 0.0
            self.ttl[dead_idx] = 0.0
            self.temperature[dead_idx] = self.surface_temp

        self.points = _sanitize_points(self.points, fallback=self.center)
        self.velocity = np.nan_to_num(self.velocity, nan=0.0, posinf=0.0, neginf=0.0)
        self.ttl = np.nan_to_num(self.ttl, nan=0.0, posinf=0.0, neginf=0.0)
        self.temperature = np.nan_to_num(self.temperature, nan=self.surface_temp, posinf=self.core_temp, neginf=self.surface_temp)
        self.mesh.points = self.points
        self._refresh_colors()

    def add_to_plotter(self, plotter):
        plotter.add_mesh(
            self.mesh,
            scalars="rgb",
            rgb=True,
            style="points",
            point_size=8.0,
            render_points_as_spheres=True,
            opacity=0.95,
            name=self.name,
        )


class Sun:
    def __init__(self, center=(10, -4, 1.5), radius=3.0, inclination=14.0, name="sun"):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.inclination = np.radians(float(inclination))
        self.name = name

        self.core_temp = 15000.0
        self.mantle_temp = 9000.0
        self.surface_temp = 5600.0

        self.rotation_phase = 0.0
        self.activity_phase = 1.7
        self.rng = np.random.default_rng(77)
        self.velocity = np.zeros(3, dtype=float)
        self.captured = False
        self.capture_center = None
        self.capture_mass = None
        self.capture_radius = None
        self.absorbed = False
        self.destruction_progress = 0.0

        self.nearest_bh_center = None
        self.nearest_bh_mass = 0.0
        self.nearest_bh_radius = 0.0
        self.nearest_bh_dist = np.inf

        self.core_radius_0 = self.radius * 0.42
        self.mantle_radius_0 = self.radius * 0.72
        self.surface_radius_0 = self.radius
        self.corona_radius_0 = self.radius * 1.18

        self.core_template = pv.Sphere(radius=1.0, theta_resolution=56, phi_resolution=56)
        self.mantle_template = pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64)
        self.surface_template = pv.Sphere(radius=1.0, theta_resolution=84, phi_resolution=84)
        self.corona_template = pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64)

        self.core_mesh = self.core_template.copy(deep=True)
        self.mantle_mesh = self.mantle_template.copy(deep=True)
        self.surface_mesh = self.surface_template.copy(deep=True)
        self.corona_mesh = self.corona_template.copy(deep=True)

        self.core_phase = self.rng.uniform(0.0, 2.0 * np.pi, size=self.core_mesh.n_points)
        self.mantle_phase = self.rng.uniform(0.0, 2.0 * np.pi, size=self.mantle_mesh.n_points)
        self.surface_phase = self.rng.uniform(0.0, 2.0 * np.pi, size=self.surface_mesh.n_points)
        self.corona_phase = self.rng.uniform(0.0, 2.0 * np.pi, size=self.corona_mesh.n_points)

        self.eruptions = SolarEruptionParticles(
            center=self.center,
            radius=self.radius,
            max_particles=3200,
            surface_temp=self.surface_temp,
            core_temp=self.core_temp,
            name=f"{self.name}_eruptions",
        )
        self.fragments = DebrisCloud(
            host_mass=1.0,
            bh_radius=1.0,
            center=self.center,
            max_particles=2600,
            name=f"{self.name}_fragments",
        )

        self._update_layers(time=0.0)
        self._update_geometry(time=0.0)

    def _translate(self, delta):
        delta = np.asarray(delta, dtype=float)
        self.center = self.center + delta
        self.eruptions.center = self.center
        self.eruptions.translate(delta)

    def _tilt(self, vec):
        x, y, z = vec
        yi = y * np.cos(self.inclination) - z * np.sin(self.inclination)
        zi = y * np.sin(self.inclination) + z * np.cos(self.inclination)
        return np.array([x, yi, zi], dtype=float)

    def _update_bh_influence(self, black_holes):
        self.nearest_bh_center = None
        self.nearest_bh_mass = 0.0
        self.nearest_bh_radius = 0.0
        self.nearest_bh_dist = np.inf
        if not black_holes:
            return

        nearest = None
        nearest_dist = np.inf
        for black_hole in black_holes:
            rel = np.asarray(black_hole.position, dtype=float) - self.center
            dist = float(np.linalg.norm(rel))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = black_hole

        if nearest is None:
            return

        self.nearest_bh_center = np.asarray(nearest.position, dtype=float).copy()
        self.nearest_bh_mass = float(nearest.mass)
        self.nearest_bh_radius = float(nearest.radius)
        self.nearest_bh_dist = nearest_dist

    def _try_capture(self):
        if self.captured or self.absorbed:
            return
        if self.nearest_bh_center is None:
            return

        # Захват начинается, когда центр солнца оказывается внутри горизонта событий.
        if self.nearest_bh_dist <= self.nearest_bh_radius:
            self.captured = True
            self.capture_center = self.nearest_bh_center.copy()
            self.capture_mass = float(self.nearest_bh_mass)
            self.capture_radius = float(self.nearest_bh_radius)

    def _update_destruction(self, dt):
        if self.absorbed:
            self.destruction_progress = 1.0
            return
        if self.nearest_bh_center is None:
            self.destruction_progress = max(0.0, self.destruction_progress - 0.03 * dt)
            return

        tidal_radius = max(self.nearest_bh_radius * 4.6, self.radius * 1.8)
        if self.nearest_bh_dist <= tidal_radius:
            proximity = np.clip(
                (tidal_radius - self.nearest_bh_dist) / (tidal_radius - self.nearest_bh_radius * 0.72 + 1e-9),
                0.0,
                1.0,
            )
            growth = (0.08 + 0.9 * proximity ** 1.4) * dt
            if self.captured:
                growth *= 2.1
            self.destruction_progress = min(
                1.0,
                max(self.destruction_progress, 0.75 * float(proximity)) + growth,
            )
        else:
            self.destruction_progress = max(0.0, self.destruction_progress - 0.05 * dt)

    def _apply_capture_motion(self, dt):
        if not self.captured or self.absorbed:
            return

        rel = self.capture_center - self.center
        dist = float(np.linalg.norm(rel))
        rel_dir = rel / (dist + 1e-9)

        core_radius = max(self.capture_radius * 0.32, 1e-3)
        accel_mag = (2.6 + 4.2 * self.capture_mass) / max(dist ** 2, core_radius ** 2)
        accel = rel_dir * accel_mag

        self.velocity = self.velocity + accel * dt
        self.velocity *= (1.0 - 0.06 * dt)
        delta = self.velocity * dt
        self._translate(delta)

        if dist <= self.capture_radius * 0.54:
            self.absorbed = True
            self.destruction_progress = 1.0
            self.velocity[:] = 0.0

    def _update_layers(self, time):
        activity = 0.55 + 0.45 * np.sin(0.7 * time + self.activity_phase)
        activity += 0.55 * self.destruction_progress

        core_norm = np.clip(0.90 + 0.08 * np.sin(1.9 * time + self.core_phase), 0.0, 1.0)
        mantle_norm = np.clip(
            0.62
            + 0.16 * np.sin(1.3 * time + self.mantle_phase)
            + 0.10 * np.sin(3.1 * time + 0.7 * self.mantle_phase)
            + 0.08 * activity,
            0.0,
            1.0,
        )
        spots = np.sin(4.0 * self.surface_phase - 2.1 * time + 1.3 * np.sin(self.surface_phase))
        surface_norm = np.clip(
            0.34 + 0.22 * np.sin(1.4 * time + self.surface_phase) + 0.16 * activity - 0.11 * spots,
            0.0,
            1.0,
        )

        self.core_mesh.point_data["temperature_norm"] = core_norm
        self.mantle_mesh.point_data["temperature_norm"] = mantle_norm
        self.surface_mesh.point_data["temperature_norm"] = surface_norm

    def _update_geometry(self, time):
        destroy = self.destruction_progress
        global_scale = np.clip(1.0 - 0.92 * destroy, 0.03, 1.0)

        core_jitter = np.clip(1.0 + destroy * (0.10 * np.sin(2.0 * time + self.core_phase)), 0.70, 1.35)
        mantle_jitter = np.clip(
            1.0 + destroy * (0.17 * np.sin(2.5 * time + self.mantle_phase) + 0.10 * np.sin(6.0 * time + 0.4 * self.mantle_phase)),
            0.55,
            1.50,
        )
        surface_jitter = np.clip(
            1.0 + destroy * (0.30 * np.sin(3.0 * time + self.surface_phase) + 0.18 * np.sin(8.0 * time + 0.6 * self.surface_phase)),
            0.45,
            1.70,
        )
        corona_jitter = np.clip(
            1.0
            + 0.06 * np.sin(1.8 * time + self.corona_phase)
            + destroy * (0.24 * np.sin(4.0 * time + self.corona_phase)),
            0.55,
            1.85,
        )

        core_points = self.center + self.core_template.points * (self.core_radius_0 * global_scale * core_jitter[:, None])
        mantle_points = self.center + self.mantle_template.points * (self.mantle_radius_0 * global_scale * mantle_jitter[:, None])
        surface_points = self.center + self.surface_template.points * (self.surface_radius_0 * global_scale * surface_jitter[:, None])
        corona_points = self.center + self.corona_template.points * (self.corona_radius_0 * global_scale * corona_jitter[:, None])

        self.core_mesh.points = _sanitize_points(core_points, fallback=self.center)
        self.mantle_mesh.points = _sanitize_points(mantle_points, fallback=self.center)
        self.surface_mesh.points = _sanitize_points(surface_points, fallback=self.center)
        self.corona_mesh.points = _sanitize_points(corona_points, fallback=self.center)

    def _spawn_disruption_fragments(self, dt, time):
        if self.destruction_progress <= 0.12:
            return
        if self.nearest_bh_center is None:
            return

        to_bh = self.nearest_bh_center - self.center
        to_bh = to_bh / (np.linalg.norm(to_bh) + 1e-9)
        burst = max(0.0, np.sin(9.0 * time + 3.2 * self.rotation_phase))
        events = int((3.0 + 17.0 * self.destruction_progress + 12.0 * burst) * dt)

        current_surface_radius = self.surface_radius_0 * np.clip(1.0 - 0.92 * self.destruction_progress, 0.03, 1.0)
        for _ in range(events):
            rnd = self.rng.normal(size=3)
            rnd = rnd / (np.linalg.norm(rnd) + 1e-9)
            normal = rnd + 1.2 * to_bh
            normal = normal / (np.linalg.norm(normal) + 1e-9)
            tangent = np.cross(normal, np.array([0.0, 0.0, 1.0], dtype=float))
            if np.linalg.norm(tangent) < 1e-8:
                tangent = np.cross(normal, np.array([0.0, 1.0, 0.0], dtype=float))
            tangent = tangent / (np.linalg.norm(tangent) + 1e-9)

            origin = self.center + normal * current_surface_radius
            emit_count = int(self.rng.integers(6, 22) * (0.8 + 1.4 * self.destruction_progress))
            self.fragments.emit(
                origin=origin,
                tangent=tangent,
                count=emit_count,
                strength=self.destruction_progress,
                spread=current_surface_radius * (0.16 + 0.95 * self.destruction_progress),
            )

    def _spawn_eruptions(self, dt, time):
        activity = float(np.clip(0.55 + 0.45 * np.sin(0.7 * time + self.activity_phase), 0.0, 1.0))
        activity = float(np.clip(activity + 0.55 * self.destruction_progress, 0.0, 1.0))
        expected_events = (5.0 + 16.0 * activity) * dt
        events = int(self.rng.poisson(expected_events))

        for _ in range(events):
            lon = self.rotation_phase + self.rng.normal(0.0, 0.45)
            lat = self.rng.normal(0.0, 0.28)

            normal_raw = np.array(
                [
                    np.cos(lat) * np.cos(lon),
                    np.cos(lat) * np.sin(lon),
                    np.sin(lat),
                ],
                dtype=float,
            )
            normal = self._tilt(normal_raw)
            normal = normal / (np.linalg.norm(normal) + 1e-9)

            tangent = np.cross(normal, np.array([0.0, 0.0, 1.0], dtype=float))
            if np.linalg.norm(tangent) < 1e-8:
                tangent = np.cross(normal, np.array([0.0, 1.0, 0.0], dtype=float))
            tangent = tangent / (np.linalg.norm(tangent) + 1e-9)

            depth_hotness = np.clip(self.rng.beta(2.2, 1.6) * (0.5 + 0.7 * activity) + 0.3 * self.destruction_progress, 0.0, 1.0)
            count = int(self.rng.integers(20, 60) * (0.7 + activity + 0.7 * self.destruction_progress))
            current_surface_radius = self.surface_radius_0 * np.clip(1.0 - 0.92 * self.destruction_progress, 0.03, 1.0)
            origin = self.center + normal * current_surface_radius

            self.eruptions.emit(
                origin=origin,
                normal=normal,
                tangent=tangent,
                count=count,
                depth_hotness=depth_hotness,
                eruption_power=activity,
                emitter_velocity=self.velocity,
            )

    def evolve(self, dt, time, black_holes=None):
        if black_holes is None:
            black_holes = []

        self._update_bh_influence(black_holes)
        self._try_capture()
        self._update_destruction(dt)
        self._apply_capture_motion(dt)
        self.rotation_phase = np.mod(self.rotation_phase + (0.35 + 0.25 * np.sin(0.4 * time)) * dt, 2.0 * np.pi)
        self._update_layers(time)
        self._update_geometry(time)

        if self.nearest_bh_center is not None:
            self.fragments.center = self.nearest_bh_center.copy()
            self.fragments.host_mass = max(0.8, self.nearest_bh_mass)
            self.fragments.bh_radius = max(self.nearest_bh_radius, 0.2)
        else:
            self.fragments.center = self.center.copy()
            self.fragments.host_mass = 0.6
            self.fragments.bh_radius = max(self.radius * 0.4, 0.2)

        self._spawn_disruption_fragments(dt, time)
        if not self.absorbed:
            self._spawn_eruptions(dt, time)
        self.eruptions.evolve(dt)
        self.fragments.evolve(dt)

    def add_to_plotter(self, plotter):
        plotter.add_mesh(
            self.core_mesh,
            scalars="temperature_norm",
            cmap="turbo",
            clim=[0.0, 1.0],
            smooth_shading=True,
            opacity=0.92,
            show_scalar_bar=False,
            name=f"{self.name}_core",
        )
        plotter.add_mesh(
            self.mantle_mesh,
            scalars="temperature_norm",
            cmap="turbo",
            clim=[0.0, 1.0],
            smooth_shading=True,
            opacity=0.60,
            show_scalar_bar=False,
            name=f"{self.name}_mantle",
        )
        plotter.add_mesh(
            self.surface_mesh,
            scalars="temperature_norm",
            cmap="turbo",
            clim=[0.0, 1.0],
            smooth_shading=True,
            opacity=0.34,
            show_scalar_bar=False,
            name=f"{self.name}_surface",
        )
        plotter.add_mesh(
            self.corona_mesh,
            color="#ffd2a8",
            smooth_shading=True,
            opacity=0.10,
            name=f"{self.name}_corona",
        )
        self.eruptions.add_to_plotter(plotter)
        self.fragments.add_to_plotter(plotter)


class Jet:
    def __init__(self, center=(0, 0, 0), direction=(0, 0, 1), length=3.0, phase_offset=0.0, name="jet"):
        self.center = np.asarray(center, dtype=float)

        raw_direction = np.asarray(direction, dtype=float)
        self.direction = raw_direction / (np.linalg.norm(raw_direction) + 1e-9)

        self.length = float(length)
        self.base_radius = self.length * 0.12
        self.phase_offset = float(phase_offset)
        self.name = name

        self.template = pv.Cone(
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            radius=1.0,
            height=1.0,
            resolution=80,
        )
        self.template_points = self.template.points.copy()
        self.mesh = self.template.copy(deep=True)
        self._update_mesh(self.base_radius, self.length)

    def _rotation_to_direction(self):
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        target = self.direction / (np.linalg.norm(self.direction) + 1e-12)
        cos_angle = float(np.clip(np.dot(z_axis, target), -1.0, 1.0))
        if cos_angle > 1.0 - 1e-10:
            return np.eye(3, dtype=float)
        if cos_angle < -1.0 + 1e-10:
            return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)

        v = np.cross(z_axis, target)
        s = np.linalg.norm(v)
        vx = np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=float,
        )
        r = np.eye(3, dtype=float) + vx + vx @ vx * ((1.0 - cos_angle) / (s ** 2 + 1e-12))
        return r

    def _update_mesh(self, radius, length):
        radius = max(float(radius), 1e-6)
        length = max(float(length), 1e-6)
        jet_center = self.center + self.direction * (length * 0.5)

        points = self.template_points.copy()
        points[:, 0] *= radius
        points[:, 1] *= radius
        points[:, 2] *= length

        rotation = self._rotation_to_direction()
        points = points @ rotation.T
        points += jet_center[None, :]
        self.mesh.points = _sanitize_points(points, fallback=jet_center)

    def pulsate(self, time, power=1.0):
        """Пульсация джета, связанная с текущим уровнем аккреции."""
        slow_breath = 1.0 + 0.18 * np.sin(2.2 * time + self.phase_offset)
        fast_mod = 1.0 + 0.08 * np.sin(5.0 * time + 0.5 * self.phase_offset)
        size_scale = np.clip(power * slow_breath * fast_mod, 0.65, 1.7)

        radius = self.base_radius * (0.55 + 0.45 * size_scale)
        length = self.length * (0.80 + 0.40 * size_scale)
        self._update_mesh(radius, length)

    def add_to_plotter(self, plotter):
        """Добавление джета на сцену."""
        plotter.add_mesh(
            self.mesh,
            color="#6ce3ff",
            opacity=0.50,
            smooth_shading=True,
            specular=0.35,
            name=self.name,
        )


class BlackHoleScene:
    def __init__(self):
        self.plotter = pv.Plotter(window_size=(1280, 720))
        self.black_holes = []
        self.suns = []
        self.time = 0.0
        self._setup_environment()

    def _setup_environment(self):
        self.plotter.set_background("#020409")
        self.plotter.enable_anti_aliasing()
        self.plotter.add_axes(line_width=1, color="white")

        light = pv.Light(position=(24, 20, 16), focal_point=(0, 0, 0), color="white", intensity=1.1)
        self.plotter.add_light(light)

    def add_black_hole(self, mass=1.0, position=(0, 0, 0)):
        """Добавление черной дыры в сцену."""
        black_hole = BlackHole(mass, position)
        self.black_holes.append(black_hole)
        black_hole.add_to_plotter(self.plotter)
        return black_hole

    def add_sun(self, radius=3.0, position=(7.2, -1.8, 0.9), inclination=14.0):
        """Добавление модели солнца с извержениями."""
        sun = Sun(center=position, radius=radius, inclination=inclination, name=f"sun_{len(self.suns)}")
        self.suns.append(sun)
        sun.add_to_plotter(self.plotter)
        return sun

    def _step_scene(self, dt):
        for black_hole in self.black_holes:
            black_hole.update(dt, self.time)
        for sun in self.suns:
            sun.evolve(dt, self.time, black_holes=self.black_holes)

        camera_distance = 20.0
        camera_x = camera_distance * np.cos(self.time * 0.24)
        camera_y = camera_distance * np.sin(self.time * 0.24)
        self.plotter.camera_position = [
            (camera_x, camera_y, 5.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        self.time += dt

    def animate(self, duration=10, fps=30, gif_duration=2.0, interactive=True, loop_forever=True):
        """Запуск анимации: GIF записывает первые секунды, рендер может идти бесконечно."""
        dt = 1.0 / fps
        total_frames = max(1, int(duration * fps))
        gif_frames = int(max(0.0, min(gif_duration, duration)) * fps)
        should_record_gif = gif_frames > 0

        if interactive:
            if should_record_gif:
                self.plotter.open_gif("black_hole_animation.gif", fps=fps)
            self.plotter.show(auto_close=False, interactive_update=True)
            frame = 0
            next_tick = time.perf_counter()

            try:
                while True:
                    if getattr(self.plotter, "_closed", False):
                        break
                    if not loop_forever and frame >= total_frames:
                        break

                    now = time.perf_counter()
                    if now < next_tick:
                        time.sleep(min(0.004, next_tick - now))
                        continue

                    self._step_scene(dt)
                    self.plotter.update()

                    if should_record_gif and frame < gif_frames:
                        self.plotter.write_frame()
                        if frame + 1 >= gif_frames and getattr(self.plotter, "mwriter", None) is not None:
                            self.plotter.mwriter.close()
                            self.plotter.mwriter = None

                    frame += 1
                    next_tick += dt
                    if (now - next_tick) > (2.0 * dt):
                        next_tick = now + dt
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"Animation loop stopped after exception: {exc}")
            finally:
                if should_record_gif and getattr(self.plotter, "mwriter", None) is not None:
                    self.plotter.mwriter.close()
                    self.plotter.mwriter = None
        else:
            if should_record_gif:
                self.plotter.open_gif("black_hole_animation.gif", fps=fps)

            for frame in range(total_frames):
                self._step_scene(dt)
                if should_record_gif and frame < gif_frames:
                    self.plotter.write_frame()

            if should_record_gif and getattr(self.plotter, "mwriter", None) is not None:
                self.plotter.mwriter.close()
                self.plotter.mwriter = None
            self.plotter.close()

    def show(self):
        """Показать статичную сцену."""
        self.plotter.camera_position = [(18, 0, 4), (0, 0, 0), (0, 0, 1)]
        self.plotter.show()


if __name__ == "__main__":
    scene = BlackHoleScene()
    scene.add_black_hole(mass=1.0, position=(0, 0, 0))
    scene.add_sun(radius=2.9, position=(7.0, -1.6, 0.8), inclination=16.0)
    scene.animate(duration=6, fps=24, gif_duration=2.0, interactive=True)
