# cython: language_level=3

import numpy as np
cimport numpy as np

def simulate(
    np.ndarray[np.float64_t, ndim=2] positions,
    np.ndarray[np.float64_t, ndim=2] velocities,
    np.ndarray[np.float64_t, ndim=2] object_vertices,
    float dt
):
    """
    Simulate wind particles moving around a stationary object.

    Parameters:
    - positions: Array of particle positions (num_particles, 3).
    - velocities: Array of particle velocities (num_particles, 3).
    - object_vertices: Array of object vertex positions (num_vertices, 3).
    - dt: Time step for the simulation.
    """
    cdef int num_particles = positions.shape[0]
    cdef int num_vertices = object_vertices.shape[0]
    cdef int i, j

    cdef double collision_distance = 0.1  # Adjust as needed
    cdef np.ndarray[np.float64_t, ndim=1] wind_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Loop over particles
    for i in range(num_particles):
        # Apply wind force
        velocities[i] += wind_direction * dt

        # Simple collision detection
        for j in range(num_vertices):
            vec_to_vertex = positions[i] - object_vertices[j]
            distance = np.linalg.norm(vec_to_vertex)
            if distance < collision_distance:
                # Reflect velocity (simple collision response)
                normal = vec_to_vertex / distance
                velocities[i] -= 2 * np.dot(velocities[i], normal) * normal
                break  # Exit loop after handling collision

        # Update position
        positions[i] += velocities[i] * dt

    return positions, velocities
