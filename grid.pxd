# grid.pxd
cimport numpy as cnp
from mathutils import Vector

cdef class MACGrid:
    cdef cnp.npy_intp grid_size[3]
    cdef double cell_size
    cdef cnp.ndarray u       # x-velocity at faces (size: nx+1, ny, nz)
    cdef cnp.ndarray v       # y-velocity at faces (size: nx, ny+1, nz)
    cdef cnp.ndarray w       # z-velocity at faces (size: nx, ny, nz+1)
    cdef cnp.ndarray pressure    # pressure at cell centers (size: nx, ny, nz)
    cdef cnp.ndarray density     # density at cell centers (size: nx, ny, nz)
    cdef cnp.ndarray position    # positions of cell centers (size: nx, ny, nz, 3)

    cpdef cnp.ndarray get_cell_position(self, cnp.npy_intp x, cnp.npy_intp y, cnp.npy_intp z)
    cpdef void update_particle_positions(self, object particle_objects, int frame)
