# grid.pyx
# cython: language_level=3

import numpy as np
cimport numpy as cnp
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.sparse.linalg import cg 
from scipy.sparse import coo_matrix

cdef class MACGrid:

    def __init__(self, tuple grid_size, double cell_size):
        cdef cnp.npy_intp x, y, z
        self.grid_size = grid_size
        self.cell_size = cell_size
        cdef cnp.npy_intp nx = self.grid_size[0]
        cdef cnp.npy_intp ny = self.grid_size[1]
        cdef cnp.npy_intp nz = self.grid_size[2]
        print(nx,ny,nz)
        self.u = np.zeros((nx+1, ny, nz), dtype=np.float64)
        self.v = np.zeros((nx, ny+1, nz), dtype=np.float64)
        self.w = np.zeros((nx, ny, nz+1), dtype=np.float64)
        self.pressure = np.zeros((nx, ny, nz), dtype=np.float64)
        self.divergence =  np.zeros((nx, ny, nz), dtype=np.float64)
        self.max_vel = 0
        self.density = np.zeros((nx, ny, nz), dtype=np.float64)
        self.solid_mask = np.zeros((nx, ny, nz), dtype=np.float32)
        self.position = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    self.position[x, y, z, :] = [(x + 0.5) * cell_size, (y + 0.5) * cell_size, (z + 0.5) * cell_size]  # Mac grid offset

    cpdef cnp.ndarray get_mask(self, object bvh_tree):
        cdef cnp.npy_intp nx = self.grid_size[0]
        cdef cnp.npy_intp ny = self.grid_size[1]
        cdef cnp.npy_intp nz = self.grid_size[2]
        cdef object pos
        cdef object direction
        cdef int intersections
        cdef object result
        cdef double max_distance
        cdef bint hit 
        self.solid_mask = np.zeros((nx, ny, nz), dtype=np.int8)
        direction = Vector((1.0, 0.0, 0.0)) 
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    pos = Vector(self.position[x, y, z, :])
                    intersections = 0
                    start = pos.copy()
                    max_distance = 1e6 
                    hit = True
                    while hit:
                        result = bvh_tree.ray_cast(start, direction, max_distance)
                        if result[0] is not None:
                            location, normal, index, distance = result
                            intersections += 1
                            start = location + direction * 1e-5
                            max_distance -= distance + 1e-5
                        else:
                            hit = False
                    if intersections % 2 == 1:
                        self.solid_mask[x, y, z] = 1  
        return self.solid_mask

    cpdef cnp.ndarray get_cell_position(self, cnp.npy_intp x, cnp.npy_intp y, cnp.npy_intp z):
        return self.position[x, y, z, :]

    cpdef void set_face_velocities(self, cnp.npy_intp x, cnp.npy_intp y, cnp.npy_intp z, cnp.ndarray vel):
        cdef cnp.npy_intp nx = self.grid_size[0]
        cdef cnp.npy_intp ny = self.grid_size[1]
        cdef cnp.npy_intp nz = self.grid_size[2]
        if 0 <= x < nx + 1:
            self.u[x, y, z] = vel[0]
        if 0 <= x - 1 < nx + 1:
            self.u[x - 1, y, z] = vel[0]    
        if 0 <= y < ny + 1:
            self.v[x, y, z] = vel[1]
        if 0 <= y - 1 < ny + 1:
            self.v[x, y - 1, z] = vel[1]
        if 0 <= z < nz + 1:
            self.w[x, y, z] = vel[2]
        if 0 <= z - 1 < nz + 1:
            self.w[x, y, z - 1] = vel[2]

    cpdef int index(self, i, j, k, nx, ny, nz):
            return i + j * nx + k * nx * ny

    cpdef object build_sparse(self):
        cdef cnp.npy_intp nx, ny, nz
        cdef list row = []
        cdef list col = []
        cdef list data = []
        cdef object laplacian
        cdef int system_index = 0
        cdef dict cell_to_sys_idx = {}
        cdef dict sys_idx_to_cell = {}
        cdef int x, y, z
        nx, ny, nz = self.grid_size

        # First pass: assign system indices to fluid cells
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if self.solid_mask[x, y, z] == 0:
                        cell_to_sys_idx[(x, y, z)] = system_index
                        sys_idx_to_cell[system_index] = (x, y, z)
                        system_index += 1
        n_points = system_index 

        # Second pass: build the matrix
        for (x, y, z), p in cell_to_sys_idx.items():
            diag = 0
            for dx, dy, dz in [(-1, 0, 0), (1, 0, 0),
                               (0, -1, 0), (0, 1, 0),
                               (0, 0, -1), (0, 0, 1)]:
                nx_, ny_, nz_ = x + dx, y + dy, z + dz
                if 0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz:
                    diag += 1
                    if self.solid_mask[nx_, ny_, nz_] == 0:
                        neighbor_p = cell_to_sys_idx[(nx_, ny_, nz_)]
                        row.append(p)
                        col.append(neighbor_p)
                        data.append(-1)
                else:
                    diag += 1  # Increment diagonal for boundary cells
            row.append(p)
            col.append(p)
            data.append(diag)
        laplacian = coo_matrix((data, (row, col)), shape=(n_points, n_points))
        return laplacian.tocsr(), cell_to_sys_idx, sys_idx_to_cell

