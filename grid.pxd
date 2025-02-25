# grid.pxd
cimport numpy as cnp
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg 
from scipy.sparse import coo_matrix

cdef class MACGrid:
    cdef cnp.npy_intp grid_size[3]
    cdef double cell_size
    cdef cnp.ndarray u       # x-vel (ux) at faces (array sz: nx+1, ny, nz)
    cdef cnp.ndarray v       # y-vel (uy) at faces (array sz: nx, ny+1, nz)
    cdef cnp.ndarray w       # z-vel (uz) at faces (array sz: nx, ny, nz+1)
    cdef double max_vel # max current velocity in grid
    cdef cnp.ndarray pressure    # pressure (p) at cell center (array sz: nx, ny, nz)
    cdef cnp.ndarray density     # density (rho) at cell center (array sz, ny, nz)
    cdef cnp.ndarray position    # positions (pos) of cell center (array sz: nx, ny, nz, 3)
    cdef cnp.ndarray solid_mask  # Array representing the mesh, for collisions
    cdef cnp.ndarray divergence  # Divergence of the velocities for pressure solving
    cdef double max_pressure
    cdef double min_pressure
    cpdef int index(self, i, j, k, nx, ny, nz) # Get linear index of cell
    cpdef cnp.ndarray get_mask(self, object bvh_tree) # Build solid object mask, at start of simulation
    cpdef cnp.ndarray get_cell_position(self, cnp.npy_intp x, cnp.npy_intp y, cnp.npy_intp z)
    cpdef void set_face_velocities(self, cnp.npy_intp x, cnp.npy_intp y, cnp.npy_intp z, cnp.ndarray vel)
    cpdef object build_sparse(self)# Build sparse matrix for pressure solving
    
'''For a grid of nx, ny, nz cells, we store the pressure in a
nx, ny, nz array, the x component of the velocity in a nx+1, ny, nz
array, the y component of the velocity in a nx, ny + 1, nz array, and
the z component of the velocity in a nx, ny, nz + 1 array.
(Fluid Simulation For Computer Graphics: A Tutorial in Grid Based and Particle
Based Methods)'''