# cython: language_level=3

import numpy as np
cimport numpy as np

def simulate(
    np.ndarray[np.float64_t, ndim=2] positions,
    np.ndarray[np.float64_t, ndim=2] velocities,
    float time_step,
    np.ndarray[np.float64_t, ndim=1] wind_direction
):
    """
    Simulate wind particles.

    Parameters:
    - positions: Array of particle positions (num_particles, 3).
    - velocities: Array of particle velocities (num_particles, 3).
    - time_step: Time step for the simulation.
    - wind_direction: Wind direction vector (3,).
    """
    cdef int num_particles = positions.shape[0]
    cdef int i

    cdef double wind_norm = np.linalg.norm(wind_direction)
    if wind_norm != 0:
        wind_direction = wind_direction / wind_norm
    else:
        wind_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    for i in range(num_particles):
        velocities[i] += wind_direction * time_step #Wind force
        positions[i] += velocities[i] * time_step 

    return positions, velocities
