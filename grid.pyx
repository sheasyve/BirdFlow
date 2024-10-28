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
        cdef cnp.npy_intp nx, ny, nz
        nx, ny, nz = self.grid_size
        pos = Vector((0.0, 0.0, 0.0)) 
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    pos.x, pos.y, pos.z = self.position[x, y, z, :]
                    location, normal, index, distance = bvh_tree.find_nearest(pos)
                    if location and (location - pos).length < self.cell_size * 0.001:
                        self.solid_mask[x, y, z] = 1

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

    cpdef void update_particle_positions(self, object particle_objects, int frame):
        positions = self.position.reshape(-1, 3)
        for i, particle in enumerate(particle_objects):
            if i < positions.shape[0]:
                particle.location = Vector(positions[i])
                particle.keyframe_insert(data_path="location", frame=frame)
      
    cpdef int index(self, i, j, k, nx, ny, nz):
            return i + j * nx + k * nx * ny

    cpdef object build_sparse(self):
        cdef cnp.npy_intp nx, ny, nz
        cdef list row = []
        cdef list col = []
        cdef list data = []
        cdef object laplacian
        nx, ny, nz = self.grid_size
        n_points = nx * ny * nz
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if self.solid_mask[x, y, z] == 1:
                        continue 
                    p = self.index(x, y, z, nx, ny, nz)
                    row.append(p)
                    col.append(p)
                    data.append(6)  # Center coefficient
                    # Check each neighbor and add connections if not solid
                    if x > 0 and self.solid_mask[x - 1, y, z] == 0:
                        row.append(p)
                        col.append(self.index(x - 1, y, z, nx, ny, nz))
                        data.append(-1)
                    if x < nx - 1 and self.solid_mask[x + 1, y, z] == 0:
                        row.append(p)
                        col.append(self.index(x + 1, y, z, nx, ny, nz))
                        data.append(-1)
                    if y > 0 and self.solid_mask[x, y - 1, z] == 0:
                        row.append(p)
                        col.append(self.index(x, y - 1, z, nx, ny, nz))
                        data.append(-1)
                    if y < ny - 1 and self.solid_mask[x, y + 1, z] == 0:
                        row.append(p)
                        col.append(self.index(x, y + 1, z, nx, ny, nz))
                        data.append(-1)
                    if z > 0 and self.solid_mask[x, y, z - 1] == 0:
                        row.append(p)
                        col.append(self.index(x, y, z - 1, nx, ny, nz))
                        data.append(-1)
                    if z < nz - 1 and self.solid_mask[x, y, z + 1] == 0:
                        row.append(p)
                        col.append(self.index(x, y, z + 1, nx, ny, nz))
                        data.append(-1)
        laplacian = coo_matrix((data, (row, col)), shape=(n_points, n_points))
        return laplacian.tocsr()
        