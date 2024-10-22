import bpy
import bmesh  # type: ignore
from mathutils import Vector  # type: ignore
from mathutils.bvhtree import BVHTree  # type: ignore
import numpy as np # type: ignore
import importlib
from . import cmain

class EulerianGrid:
    def __init__(self, grid_size, cell_size):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.cells = [[[GridCell(x, y, z) for z in range(self.grid_size[2])] for y in range(self.grid_size[1])] for x in range(self.grid_size[0])]

    def simulate(self, wind, dt):
        # Apply wind as a force to all grid cells
        self.apply_forces(wind, dt)
        
        # Iterate through all cells and update their positions based on their velocities
        for cell in self.iterate_cells():
            cell.position += cell.velocity * dt
            print(f"Simulating: Position {cell.position}, Velocity {cell.velocity}")

    def apply_forces(self, wind, dt):
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    self.cells[x][y][z].apply_force(wind * dt)
    
    def update_particle_positions(self, particle_objects, frame):
        positions = [cell.position for cell in self.iterate_cells()]  # Example way of generating positions
        for i, particle in enumerate(particle_objects):
            particle.location = positions[i]
            particle.keyframe_insert(data_path="location", frame=frame)

    def collide(self, bvh_tree, dt):
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    cell = self.cells[x][y][z]
                    origin = cell.position - cell.velocity * dt
                    velocity_norm = np.linalg.norm(cell.velocity)
                    if velocity_norm == 0:
                        continue
                    direction = Vector(cell.velocity / velocity_norm)
                    location, normal, _, distance = bvh_tree.ray_cast(Vector(origin), direction, velocity_norm * dt)
                    if location is not None and distance <= velocity_norm * dt:
                        cmain.redirect_grid(self, np.array(location, dtype=np.float64), np.array(normal, dtype=np.float64), x, y, z)

    def iterate_cells(self):
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    yield self.cells[x][y][z]

class GridCell:
    def __init__(self, x, y, z):
        self.position = np.array([x, y, z], dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)

    def apply_force(self, force):
        self.velocity += force


