# cython: language_level=3

import numpy as np
cimport numpy as np

def simulate(
    np.ndarray[np.float64_t, ndim=2] positions,
    np.ndarray[np.float64_t, ndim=2] velocities,
    float dt
):
    """
    Simulate wind particles moving without collision detection.

    Parameters:
    - positions: Array of particle positions (num_particles, 3).
    - velocities: Array of particle velocities (num_particles, 3).
    - dt: Time step for the simulation.
    """
    cdef int num_particles = positions.shape[0]
    cdef int i

    cdef np.ndarray[np.float64_t, ndim=1] wind_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    for i in range(num_particles):
        # Apply wind force
        velocities[i] += wind_direction * dt
        # Update position
        positions[i] += velocities[i] * dt

    return positions, velocities

