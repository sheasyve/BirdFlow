# cython: language_level=3

import numpy as np
cimport numpy as np

def simulate(np.ndarray[np.float64_t, ndim=2] velocity, np.ndarray[np.float64_t, ndim=2] pressure, double dt):
    cdef int i, j
    cdef int size_x, size_y
    size_x = velocity.shape[0]
    size_y = velocity.shape[1]

    for i in range(size_x):
        for j in range(size_y):
            velocity[i, j] += -0.5 * pressure[i, j] * dt

    return velocity
