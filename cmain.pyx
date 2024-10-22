# cython: language_level=3

import numpy as np
cimport numpy as cnp
from mathutils import Vector 

cdef class GridCell:
    cdef public cnp.ndarray position
    cdef public cnp.ndarray velocity

    def __init__(self, double x, double y, double z):
        self.position = np.array([x, y, z], dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)

    def apply_force(self, cnp.ndarray force):
        if self.velocity is None or not isinstance(self.velocity, cnp.ndarray):
            self.velocity = np.zeros(3, dtype=np.float64)
        self.velocity += force

cdef class EulerianGrid:
    cdef int grid_size[3]
    cdef list cells

    def __init__(self, tuple grid_size, double cell_size):
        self.grid_size = (grid_size[0], grid_size[1], grid_size[2])
        self.cells = [[[GridCell(x * cell_size, y * cell_size, z * cell_size) 
                        for z in range(self.grid_size[2])]
                        for y in range(self.grid_size[1])]
                        for x in range(self.grid_size[0])]

    cpdef list iterate_cells(self):
        cdef list cell_list = []
        cdef int x, y, z
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    cell_list.append(self.cells[x][y][z])
        return cell_list

    cpdef GridCell get_cell(self, int x, int y, int z):
        return self.cells[x][y][z]

    cpdef list get_cells(self):
        return self.cells

    def simulate(self, cnp.ndarray wind, double dt):
        cy_simulate(self, wind, dt)

    def collide(self, object bvh_tree, double dt):
        cy_collide(self, bvh_tree, dt)

    def update_particle_positions(self, object particle_objects, int frame):
        positions = [cell.position for cell in self.iterate_cells()]
        for i, particle in enumerate(particle_objects):
            if i < len(positions):
                particle.location = positions[i]
                particle.keyframe_insert(data_path="location", frame=frame)

cpdef void cy_apply_forces(EulerianGrid grid, cnp.ndarray wind, double dt):
    #Simulate the wind force
    cdef int x, y, z
    cdef cnp.ndarray wind_force = wind * dt
    for x in range(grid.grid_size[0]):
        for y in range(grid.grid_size[1]):
            for z in range(grid.grid_size[2]):
                grid.get_cell(x, y, z).apply_force(wind_force)

cpdef void redirect_grid(
    object eulerian_grid,
    cnp.ndarray[cnp.float64_t, ndim=1] location,
    cnp.ndarray[cnp.float64_t, ndim=1] normal,
    int x,
    int y,
    int z
):
    #Collision redirect
    cdef cnp.ndarray[cnp.float64_t, ndim=1] velocity, reflected_velocity
    cdef double dot_product
    cell = eulerian_grid.get_cell(x, y, z)
    velocity = cell.velocity
    dot_product = np.dot(velocity, normal)
    reflected_velocity = velocity - 2 * dot_product * normal
    damping_factor = 0.8
    cell.velocity[:] = reflected_velocity * damping_factor
    if np.dot(velocity, normal) < 0:
        cell.velocity[:] = np.zeros(3, dtype=np.float64)  # No-slip boundary condition

cpdef void cy_collide(EulerianGrid grid, object bvh_tree, double dt):
    #Check collisions
    cdef int x, y, z
    cdef GridCell cell
    cdef object origin, direction
    cdef double velocity_norm
    cdef object location, normal, distance
    for x in range(grid.grid_size[0]):
        for y in range(grid.grid_size[1]):
            for z in range(grid.grid_size[2]):
                cell = grid.get_cell(x, y, z)
                origin = Vector(cell.position - cell.velocity * dt)
                velocity_norm = np.linalg.norm(cell.velocity)
                if velocity_norm == 0:
                    continue 
                direction = Vector(cell.velocity / velocity_norm)
                location, normal, _, distance = bvh_tree.ray_cast(origin, direction, velocity_norm * dt)
                if location is not None and distance <= velocity_norm * dt:
                    redirect_grid(grid, np.array(location, dtype=np.float64), np.array(normal, dtype=np.float64), x, y, z)

cpdef void cy_simulate(EulerianGrid grid, cnp.ndarray wind, double dt):
    #Simulation main
    cy_apply_forces(grid, wind, dt)
    cdef GridCell cell
    for cell in grid.iterate_cells():
        cell.position += cell.velocity * dt
    '''More realistic workflow
    # Step 1: Apply wind
    cy_apply_forces(grid, wind, dt)
    # Step 2: Diffuse (Viscosity)
    cy_diffuse(grid, dt, viscosity=0.01)
    # Step 3: Collision Handling
    cy_collide(grid, bvh_tree, dt)  # Detect and handle collisions
    # Step 4: Advect fluid properties across the grid
    cy_advect(grid, dt)
    # Step 5: Pressure solve to ensure incompressibility
    cy_solve_pressure(grid, dt, iterations=20)
    # Step 6: Project velocity field
    cy_project(grid) '''
