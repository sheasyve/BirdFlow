# cython: language_level=3

import numpy as np
cimport numpy as cnp
from mathutils import Vector  # Assuming you are using this for Vector operations

def simulate(
    cnp.ndarray[cnp.float64_t, ndim=2] positions,
    cnp.ndarray[cnp.float64_t, ndim=2] velocities,
    float time_step,
    cnp.ndarray[cnp.float64_t, ndim=1] wind_direction
):
    cdef int num_particles = positions.shape[0]
    cdef int i

    cdef double wind_norm = np.linalg.norm(wind_direction)
    if wind_norm != 0:
        wind_direction = wind_direction / wind_norm
    else:
        wind_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    for i in range(num_particles):
        velocities[i] += wind_direction * time_step  # Wind force
        positions[i] += velocities[i] * time_step 

    return positions, velocities


cpdef void redirect_grid(
    object eulerian_grid,
    cnp.ndarray[cnp.float64_t, ndim=1] location,
    cnp.ndarray[cnp.float64_t, ndim=1] normal,
    int x,
    int y,
    int z
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] velocity, reflected_velocity
    cdef double dot_product
    cell = eulerian_grid.cells[x][y][z]
    velocity = cell.velocity
    dot_product = np.dot(velocity, normal)
    reflected_velocity = velocity - 2 * dot_product * normal
    cell.velocity[:] = reflected_velocity * 0.8  
