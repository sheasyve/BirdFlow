# cmain.pyx
# cython: language_level=3
import numpy as np
cimport numpy as cnp
from mathutils import Vector
from grid cimport MACGrid
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg 

cdef enum Component:
    U = 0
    V = 1
    W = 2
    S = -1

# -- Interpolation --

cdef void get_index_and_offset(double[:] pos, double cell_size, cnp.npy_intp *field_shape,
                               cnp.npy_intp* i, cnp.npy_intp* j, cnp.npy_intp* k,
                               double* fx, double* fy, double* fz, Component component):
    # Get i, j, k and fx, fy, fz for interpolation, from position
    cdef double x = pos[0] / cell_size
    cdef double y = pos[1] / cell_size
    cdef double z = pos[2] / cell_size
    if component == Component.U:
        i[0] = <cnp.npy_intp>x
        j[0] = <cnp.npy_intp>(y - 0.5)
        k[0] = <cnp.npy_intp>(z - 0.5)
        fx[0] = x - i[0]
        fy[0] = (y - 0.5) - j[0]
        fz[0] = (z - 0.5) - k[0]
    elif component == Component.V:
        i[0] = <cnp.npy_intp>(x - 0.5)
        j[0] = <cnp.npy_intp>y
        k[0] = <cnp.npy_intp>(z - 0.5)
        fx[0] = (x - 0.5) - i[0]
        fy[0] = y - j[0]
        fz[0] = (z - 0.5) - k[0]
    elif component == Component.W:
        i[0] = <cnp.npy_intp>(x - 0.5)
        j[0] = <cnp.npy_intp>(y - 0.5)
        k[0] = <cnp.npy_intp>z
        fx[0] = (x - 0.5) - i[0]
        fy[0] = (y - 0.5) - j[0]
        fz[0] = z - k[0]
    else: # Scalar field
        i[0] = <cnp.npy_intp>x
        j[0] = <cnp.npy_intp>y
        k[0] = <cnp.npy_intp>z
        fx[0] = x - i[0]
        fy[0] = y - j[0]
        fz[0] = z - k[0]
    # Clamp Indexes
    if i[0] < 0:
        i[0] = 0
    elif i[0] > field_shape[0] - 2:
        i[0] = field_shape[0] - 2
    if j[0] < 0:
        j[0] = 0
    elif j[0] > field_shape[1] - 2:
        j[0] = field_shape[1] - 2
    if k[0] < 0:
        k[0] = 0
    elif k[0] > field_shape[2] - 2:
        k[0] = field_shape[2] - 2
    # Clamp Offsets
    fx[0] = max(0.0, min(fx[0], 1.0))
    fy[0] = max(0.0, min(fy[0], 1.0))
    fz[0] = max(0.0, min(fz[0], 1.0))

cpdef double interp_dir(double u1, double u2, double fdir):
    return (u1 * (1 - fdir) + u2 * fdir) 

cpdef double trilinear_interpolate(cnp.ndarray face_u, cnp.npy_intp i, cnp.npy_intp j, cnp.npy_intp k, double fx, double fy, double fz):
    '''Main trilinear interpolation function, find velocity of a component (u, v, w)
    Grid cell index: (i, j, k) , Offset from grid cell: (fx, fy, fz)'''
    cdef double u000, u100, u010, u110, u001, u101, u011, u111
    # Grid dims
    cdef cnp.npy_intp n0 = face_u.shape[0]
    cdef cnp.npy_intp n1 = face_u.shape[1]
    cdef cnp.npy_intp n2 = face_u.shape[2]
    # Velocity of each face
    u000 = face_u[i, j, k]
    u100 = face_u[min(i+1, n0-1), j, k]#i+1
    u010 = face_u[i, min(j+1, n1-1), k]#j+1
    u110 = face_u[min(i+1, n0-1), min(j+1, n1-1), k]#i+1, j+1
    u001 = face_u[i, j, min(k+1, n2-1)]#k+1
    u101 = face_u[min(i+1, n0-1), j, min(k+1, n2-1)]#i+1, k+1
    u011 = face_u[i, min(j+1, n1-1), min(k+1, n2-1)]#j+1, k+1
    u111 = face_u[min(i+1, n0-1), min(j+1, n1-1), min(k+1, n2-1)]#i+1, j+1, k+1
    # Interpolate x
    cdef double c00 = interp_dir(u000, u100, fx)
    cdef double c01 = interp_dir(u001, u101, fx)
    cdef double c10 = interp_dir(u010, u110, fx)
    cdef double c11 = interp_dir(u011, u111, fx)
    # Interpolate y
    cdef double c0 = interp_dir(c00, c10, fy)
    cdef double c1 = interp_dir(c01, c11, fy)
    # Interpolate z
    cdef double c = interp_dir(c0, c1, fz)
    return c

cpdef cnp.ndarray interp_u_at_p(MACGrid grid, cnp.ndarray pos):
    cdef cnp.ndarray[double, ndim=1] vel = np.zeros(3, dtype=np.float64)
    vel[0] = interp_component_u_at_p(grid.u, pos, Component.U, grid)
    vel[1] = interp_component_u_at_p(grid.v, pos, Component.V, grid)
    vel[2] = interp_component_u_at_p(grid.w, pos, Component.W, grid)
    return vel

cpdef double interp_component_u_at_p(cnp.ndarray face_vel, cnp.ndarray pos, Component component, MACGrid grid):
    # Interpolate one component (u, v, w)
    cdef cnp.npy_intp i = 0, j = 0, k = 0  
    cdef double fx = 0.0, fy = 0.0, fz = 0.0 
    get_index_and_offset(pos, grid.cell_size, &face_vel.shape[0], &i, &j, &k, &fx, &fy, &fz, component)
    return trilinear_interpolate(face_vel, i, j, k, fx, fy, fz)

cpdef double interp_scalar_u_at_p(cnp.ndarray scalar_field, cnp.ndarray pos, double cell_size):
    # Interpolate a scalar (i.e pressure)
    cdef cnp.npy_intp i = 0, j = 0, k = 0 
    cdef double fx = 0.0, fy = 0.0, fz = 0.0 
    get_index_and_offset(pos, cell_size, &scalar_field.shape[0], &i, &j, &k, &fx, &fy, &fz, Component.S)
    return trilinear_interpolate(scalar_field, i, j, k, fx, fy, fz)

''' -- Simulation Functions -- '''

# -- Calculate DT

cpdef double calc_dt(MACGrid grid, double initial_dt, double cell_size, cnp.ndarray wind_acceleration):
    # Calculate dt based on max velocity from pressure solve and cfl
    cdef double cfl = 0.2
    cdef double force = np.linalg.norm(wind_acceleration)
    cdef double umax = grid.max_vel + (cell_size * force) ** 0.5
    return cfl * (cell_size / umax) if umax != 0 else cell_size

# -- Apply wind force --

cpdef void initialize_velocity(MACGrid grid, double noise_magnitude=0.01):
    cdef cnp.npy_intp nx, ny, nz
    nx, ny, nz = grid.grid_size
    grid.u += (np.random.rand(nx + 1, ny, nz) - 0.5) * noise_magnitude
    grid.v += (np.random.rand(nx, ny + 1, nz) - 0.5) * noise_magnitude
    grid.w += (np.random.rand(nx, ny, nz + 1) - 0.5) * noise_magnitude

# -- Velocity Advection --

cpdef void wind_force(MACGrid grid, double dt, cnp.ndarray[double, ndim=1] wind_speed, cnp.ndarray[double, ndim=1] wind_acceleration, double damping_factor):
    # Apply wind force on grid
    cdef cnp.npy_intp x, y, z
    cdef cnp.npy_intp nx = grid.grid_size[0]
    cdef cnp.npy_intp ny = grid.grid_size[1]
    cdef cnp.npy_intp nz = grid.grid_size[2]
    for x in range(nx + 1):
        for y in range(ny):
            for z in range(nz):
                grid.u[x, y, z] *= damping_factor
                if grid.u[x, y, z] < wind_speed[0]:
                    grid.u[x, y, z] += wind_acceleration[0] * dt
                grid.u[x, y, z] = max(min(grid.u[x, y, z], wind_speed[0]), -wind_speed[0])
    for x in range(nx):
        for y in range(ny + 1):
            for z in range(nz):
                grid.v[x, y, z] *= damping_factor
                if grid.v[x, y, z] < wind_speed[1]:
                    grid.v[x, y, z] += wind_acceleration[1] * dt
                grid.v[x, y, z] = max(min(grid.v[x, y, z], wind_speed[1]), -wind_speed[1])
    for x in range(nx):
        for y in range(ny):
            for z in range(nz + 1):
                grid.w[x, y, z] *= damping_factor
                if grid.w[x, y, z] < wind_speed[2]:
                    grid.w[x, y, z] += wind_acceleration[2] * dt
                grid.w[x, y, z] = max(min(grid.w[x, y, z], wind_speed[2]), -wind_speed[2])

cpdef void advect_velocities(MACGrid grid, double dt):
    # Apply new velocities from collision with RK3 (Runge Kutta Order-3) method
    cdef cnp.ndarray uk1, vk1, wk1, uk2, vk2, wk2, uk3, vk3, wk3
    cdef cnp.ndarray new_u, new_v, new_w
    cdef cnp.npy_intp nx, ny, nz, x, y, z
    cdef cnp.ndarray pos, temp_pos
    cdef double cell_size = grid.cell_size
    nx, ny, nz = grid.grid_size
    uk1, vk1, wk1 = np.zeros_like(grid.u), np.zeros_like(grid.v), np.zeros_like(grid.w)
    uk2, vk2, wk2 = np.zeros_like(grid.u), np.zeros_like(grid.v), np.zeros_like(grid.w)
    uk3, vk3, wk3 = np.zeros_like(grid.u), np.zeros_like(grid.v), np.zeros_like(grid.w)
    new_u, new_v, new_w = np.zeros_like(grid.u), np.zeros_like(grid.v), np.zeros_like(grid.w)
    pos, temp_pos = np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    for x in range(nx + 1):
        for y in range(ny):
            for z in range(nz):
                pos[0] = x * cell_size
                pos[1] = (y + 0.5) * cell_size
                pos[2] = (z + 0.5) * cell_size
                uk1[x, y, z] = interp_component_u_at_p(grid.u, pos, Component.U, grid)
                temp_pos[:] = pos + 0.5 * dt * uk1[x, y, z]
                temp_pos[:] = np.clip(temp_pos, [0, 0, 0], [(nx - 1) * cell_size, (ny - 1) * cell_size, (nz - 1) * cell_size])
                uk2[x, y, z] = interp_component_u_at_p(grid.u, temp_pos, Component.U, grid)
                temp_pos[:] = pos + 0.75 * dt * uk2[x, y, z]
                temp_pos[:] = np.clip(temp_pos, [0, 0, 0], [(nx - 1) * cell_size, (ny - 1) * cell_size, (nz - 1) * cell_size])
                uk3[x, y, z] = interp_component_u_at_p(grid.u, temp_pos, Component.U, grid)
                new_u[x, y, z] = grid.u[x, y, z] + (2 / 9) * dt * uk1[x, y, z] + (3 / 9) * dt * uk2[x, y, z] + (4 / 9) * dt * uk3[x, y, z]
    for x in range(nx):
        for y in range(ny + 1):
            for z in range(nz):
                pos[0] = (x + 0.5) * cell_size
                pos[1] = y * cell_size
                pos[2] = (z + 0.5) * cell_size
                vk1[x, y, z] = interp_component_u_at_p(grid.v, pos, Component.V, grid)
                temp_pos[:] = pos + 0.5 * dt * vk1[x, y, z]
                temp_pos[:] = np.clip(temp_pos, [0, 0, 0], [(nx - 1) * cell_size, (ny - 1) * cell_size, (nz - 1) * cell_size])
                vk2[x, y, z] = interp_component_u_at_p(grid.v, temp_pos, Component.V, grid)
                temp_pos[:] = pos + 0.75 * dt * vk2[x, y, z]
                temp_pos[:] = np.clip(temp_pos, [0, 0, 0], [(nx - 1) * cell_size, (ny - 1) * cell_size, (nz - 1) * cell_size])
                vk3[x, y, z] = interp_component_u_at_p(grid.v, temp_pos, Component.V, grid)
                new_v[x, y, z] = grid.v[x, y, z] + (2 / 9) * dt * vk1[x, y, z] + (3 / 9) * dt * vk2[x, y, z] + (4 / 9) * dt * vk3[x, y, z]
    for x in range(nx):
        for y in range(ny):
            for z in range(nz + 1):
                pos[0] = (x + 0.5) * cell_size
                pos[1] = (y + 0.5) * cell_size
                pos[2] = z * cell_size
                wk1[x, y, z] = interp_component_u_at_p(grid.w, pos, Component.W, grid)
                temp_pos[:] = pos + 0.5 * dt * wk1[x, y, z]
                temp_pos[:] = np.clip(temp_pos, [0, 0, 0], [(nx - 1) * cell_size, (ny - 1) * cell_size, (nz - 1) * cell_size])
                wk2[x, y, z] = interp_component_u_at_p(grid.w, temp_pos, Component.W, grid)
                temp_pos[:] = pos + 0.75 * dt * wk2[x, y, z]
                temp_pos[:] = np.clip(temp_pos, [0, 0, 0], [(nx - 1) * cell_size, (ny - 1) * cell_size, (nz - 1) * cell_size])
                wk3[x, y, z] = interp_component_u_at_p(grid.w, temp_pos, Component.W, grid)
                new_w[x, y, z] = grid.w[x, y, z] + (2 / 9) * dt * wk1[x, y, z] + (3 / 9) * dt * wk2[x, y, z] + (4 / 9) * dt * wk3[x, y, z]
    grid.u[:, :, :] = new_u
    grid.v[:, :, :] = new_v
    grid.w[:, :, :] = new_w
    v_boundary_conditions(grid)

# -- Pressure Solve Utilities --

cpdef double get_vel(MACGrid grid, int x, int y, int z):
    cdef double u, v, w
    u = interp_dir(grid.u[x, y, z], grid.u[x-1, y, z], 0.5) if x > 0 else grid.u[x, y, z]
    v = interp_dir(grid.v[x, y, z], grid.v[x, y-1, z], 0.5) if y > 0 else grid.v[x, y, z]
    w = interp_dir(grid.w[x, y, z], grid.w[x, y, z-1], 0.5) if z > 0 else grid.w[x, y, z]
    return (u**2 + v**2 + w**2)**0.5

cpdef void p_boundary_conditions(MACGrid grid):
    # Neumann solid boundary conditions for pressure
    cdef cnp.npy_intp nx = grid.grid_size[0]
    cdef cnp.npy_intp ny = grid.grid_size[1]
    cdef cnp.npy_intp nz = grid.grid_size[2]
    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            for z in range(1, nz - 1):
                if grid.solid_mask[x-1, y, z] == 1 and grid.solid_mask[x, y, z] == 0:
                    grid.pressure[x, y, z] = grid.pressure[x+1, y, z]
                if grid.solid_mask[x+1, y, z] == 1 and grid.solid_mask[x, y, z] == 0:
                    grid.pressure[x, y, z] = grid.pressure[x-1, y, z]
                if grid.solid_mask[x, y-1, z] == 1 and grid.solid_mask[x, y, z] == 0:
                    grid.pressure[x, y, z] = grid.pressure[x, y+1, z]
                if grid.solid_mask[x, y+1, z] == 1 and grid.solid_mask[x, y, z] == 0:
                    grid.pressure[x, y, z] = grid.pressure[x, y-1, z]
                if grid.solid_mask[x, y, z-1] == 1 and grid.solid_mask[x, y, z] == 0:
                    grid.pressure[x, y, z] = grid.pressure[x, y, z+1]
                if grid.solid_mask[x, y, z+1] == 1 and grid.solid_mask[x, y, z] == 0:
                    grid.pressure[x, y, z] = grid.pressure[x, y, z-1]

cpdef void v_boundary_conditions(MACGrid grid):
    # No slip solid boundary conditions for velocity 
    cdef cnp.npy_intp nx = grid.grid_size[0]
    cdef cnp.npy_intp ny = grid.grid_size[1]
    cdef cnp.npy_intp nz = grid.grid_size[2]
    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            for z in range(1, nz - 1):
                if grid.solid_mask[x-1, y, z] == 1 and grid.solid_mask[x, y, z] == 0:
                    grid.u[x, y, z] = 0.0
                if grid.solid_mask[x+1, y, z] == 1 and grid.solid_mask[x, y, z] == 0:
                    grid.u[x, y, z] = 0.0
                if grid.solid_mask[x, y-1, z] == 1 and grid.solid_mask[x, y, z] == 0:
                    grid.v[x, y, z] = 0.0
                if grid.solid_mask[x, y+1, z] == 1 and grid.solid_mask[x, y, z] == 0:
                    grid.v[x, y, z] = 0.0
                if grid.solid_mask[x, y, z-1] == 1 and grid.solid_mask[x, y, z] == 0:
                    grid.w[x, y, z] = 0.0
                if grid.solid_mask[x, y, z+1] == 1 and grid.solid_mask[x, y, z] == 0:
                    grid.w[x, y, z] = 0.0
    # Handle boundaries at the edges of the grid
    for y in range(ny):
        for z in range(nz):
            grid.u[0, y, z] = 0.0  
            grid.u[nx-1, y, z] = 0.0  
    for x in range(nx):
        for z in range(nz):
            grid.v[x, 0, z] = 0.0  
            grid.v[x, ny-1, z] = 0.0  
    for x in range(nx):
        for y in range(ny):
            grid.w[x, y, 0] = 0.0  
            grid.w[x, y, nz-1] = 0.0  

# -- Pressure Solve --

cpdef void pressure_solve(MACGrid grid, double dt, int iterations=50):
    # The backbone of the fluid behavior
    cdef cnp.npy_intp nx, ny, nz
    cdef double h = grid.cell_size
    cdef double max_vel = 0
    cdef cnp.ndarray rhs
    cdef int info = 0
    cdef object A, cell_to_sys_idx, sys_idx_to_cell
    cdef int x, y, z, p
    nx, ny, nz = grid.grid_size
    A, cell_to_sys_idx, sys_idx_to_cell = grid.build_sparse()
    n_points = A.shape[0]
    rhs = np.zeros(n_points, dtype=np.float64)
    for (x, y, z), p in cell_to_sys_idx.items():
        u_diff = grid.u[x+1, y, z] - grid.u[x, y, z]
        v_diff = grid.v[x, y+1, z] - grid.v[x, y, z]
        w_diff = grid.w[x, y, z+1] - grid.w[x, y, z]
        rhs[p] = -(u_diff + v_diff + w_diff) / h
        max_vel = max(max_vel, get_vel(grid, x, y, z))
    grid.max_vel = max_vel
    fixed_cell = (0, 0, 0)
    if fixed_cell in cell_to_sys_idx:
        fixed_p = cell_to_sys_idx[fixed_cell]
        A.data[A.indptr[fixed_p]:A.indptr[fixed_p+1]] = 0
        A[fixed_p, fixed_p] = 1
        rhs[fixed_p] = 0
    pressure_solution, info = cg(A, rhs, maxiter=iterations)
    if info != 0:
        print(f"Warning: Did not converge (info={info})")
    grid.pressure[:, :, :] = 0.0  
    for p, (x, y, z) in sys_idx_to_cell.items():
        grid.pressure[x, y, z] = pressure_solution[p]
    p_boundary_conditions(grid)

# -- Pressure Projection --

cpdef void project(MACGrid grid):
    cdef int x, y, z
    cdef cnp.npy_intp nx, ny, nz
    nx, ny, nz = grid.grid_size
    cdef double h = grid.cell_size
    for x in range(nx + 1):
        for y in range(ny):
            for z in range(nz):
                if 0 < x < nx and (grid.solid_mask[x-1, y, z] == 0 or grid.solid_mask[x, y, z] == 0):
                    grid.u[x, y, z] -= (grid.pressure[x, y, z] - grid.pressure[x-1, y, z]) / h
    for x in range(nx):
        for y in range(ny + 1):
            for z in range(nz):
                if 0 < y < ny and (grid.solid_mask[x, y-1, z] == 0 or grid.solid_mask[x, y, z] == 0):
                    grid.v[x, y, z] -= (grid.pressure[x, y, z] - grid.pressure[x, y-1, z]) / h
    for x in range(nx):
        for y in range(ny):
            for z in range(nz + 1):
                if 0 < z < nz and (grid.solid_mask[x, y, z-1] == 0 or grid.solid_mask[x, y, z] == 0):
                    grid.w[x, y, z] -= (grid.pressure[x, y, z] - grid.pressure[x, y, z-1]) / h
    v_boundary_conditions(grid)

# -- Density Advection --

cpdef void advect_density(MACGrid grid, double dt):
    # Advect density scalar with RK3 method
    cdef cnp.npy_intp nx = grid.grid_size[0]
    cdef cnp.npy_intp ny = grid.grid_size[1]
    cdef cnp.npy_intp nz = grid.grid_size[2]
    cdef cnp.npy_intp x, y, z
    cdef double cell_size = grid.cell_size
    cdef cnp.ndarray new_density
    cdef cnp.ndarray pos = np.zeros(3, dtype=np.float64)
    cdef cnp.ndarray temp_pos = np.zeros(3, dtype=np.float64)
    cdef cnp.ndarray k1 = np.zeros(3, dtype=np.float64)
    cdef cnp.ndarray k2 = np.zeros(3, dtype=np.float64)
    cdef cnp.ndarray k3 = np.zeros(3, dtype=np.float64)
    new_density = np.zeros_like(grid.density)
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                pos[:] = grid.get_cell_position(x, y, z)
                k1[:] = interp_u_at_p(grid, pos)
                temp_pos[:] = pos + 0.5 * dt * k1
                k2[:] = interp_u_at_p(grid, temp_pos)
                temp_pos[:] = pos + 0.75 * dt * k2
                k3[:] = interp_u_at_p(grid, temp_pos)
                temp_pos[:] = pos + (2/9) * dt * k1 + (3/9) * dt * k2 + (4/9) * dt * k3
                new_density[x, y, z] = interp_scalar_u_at_p(grid.density, temp_pos, cell_size)
    grid.density[:, :, :] = new_density

# -- Collisions for Particles --

cpdef void redirect_particle_velocity(cnp.ndarray vel, cnp.ndarray normal, double damping_factor, double friction):
    # Reflect damped velocity based on collision normal 
    cdef cnp.ndarray reflected_u = np.zeros(3, dtype=np.float64)
    cdef cnp.ndarray tangent_u = np.zeros(3, dtype=np.float64)
    reflected_u[:] = (vel - 2 * np.dot(vel, normal) * normal) * .95
    tangent_u[:] = reflected_u - np.dot(reflected_u, normal) * normal
    reflected_u[:] -= friction * tangent_u
    vel[:] = reflected_u

cpdef cnp.ndarray collide(cnp.ndarray particle_objects, object bvh_tree, double dt, double damping_factor, double friction):
    # Collide particles with the object, if needed after simulating from grid.
    cdef int p
    cdef cnp.ndarray pos = np.zeros(3, dtype=np.float64)
    cdef cnp.ndarray vel = np.zeros(3, dtype=np.float64)
    cdef object location, normal, distance
    for p in range(particle_objects.shape[0]):
        pos[:] = particle_objects[p, :3]
        vel[:] = particle_objects[p, 3:6]
        if np.linalg.norm(vel) == 0:
            continue 
        origin = Vector(pos - vel * dt)
        direction = Vector(vel / (np.linalg.norm(vel) + .001))
        result = bvh_tree.ray_cast(origin, direction, np.linalg.norm(vel) * dt)
        if result is not None:
            location, normal, index, distance = result 
            if distance is not None and distance > 0 and distance <= np.linalg.norm(vel) * dt:
                redirect_particle_velocity(vel, np.array(normal, dtype=np.float64), damping_factor, friction)
                particle_objects[p, 3:6] = vel
                pos += vel * dt
                particle_objects[p, :3] = pos  
    return particle_objects

# -- Particle Advection

cpdef cnp.ndarray advect_particles(MACGrid grid, cnp.ndarray particle_objects, double dt, int update_p):
    cdef cnp.ndarray pos = np.zeros(3, dtype=np.float64)
    cdef cnp.ndarray temp_pos = np.zeros(3, dtype=np.float64)
    cdef cnp.ndarray k1 = np.zeros(3, dtype=np.float64)
    cdef cnp.ndarray k2 = np.zeros(3, dtype=np.float64)
    cdef cnp.ndarray k3 = np.zeros(3, dtype=np.float64)
    cdef double cell_size = grid.cell_size
    cdef cnp.npy_intp nx, ny, nz
    nx, ny, nz = grid.grid_size
    for p in range(particle_objects.shape[0]):
        pos[:] = particle_objects[p, :3]
        k1[:] = interp_u_at_p(grid, pos)
        temp_pos[:] = pos + 0.5 * dt * k1
        k2[:] = interp_u_at_p(grid, temp_pos)
        temp_pos[:] = pos + 0.75 * dt * k2
        k3[:] = interp_u_at_p(grid, temp_pos)
        new_vel = (2 / 9) * k1 + (3 / 9) * k2 + (4 / 9) * k3
        particle_objects[p, 3:6] = new_vel  
        pos += new_vel * dt
        if update_p == 1:
            particle_objects[p, :3] = pos  
    return particle_objects

# -- Simulation main --

cpdef MACGrid cy_simulate(MACGrid grid, cnp.ndarray wind, double initial_dt, 
        object bvh_tree, cnp.ndarray wind_speed, cnp.ndarray wind_acceleration, 
        double damping_factor, double cell_size, double friction):
    cdef double t = 0.0
    cdef double tframe = 1.0
    cdef double dt
    cdef cnp.npy_intp nx = grid.grid_size[0]
    cdef cnp.npy_intp ny = grid.grid_size[1]
    cdef cnp.npy_intp nz = grid.grid_size[2]
    grid.divergence = np.zeros((nx, ny, nz), dtype=np.float64)
    while t < tframe:
        dt = min(calc_dt(grid, initial_dt, cell_size, wind_acceleration), tframe - t)
        wind_force(grid, dt, wind_speed, wind_acceleration, damping_factor)
        advect_velocities(grid, dt)  
        pressure_solve(grid, dt, iterations=50)
        project(grid)
        advect_velocities(grid, dt)  
        advect_density(grid, dt)
        t += dt
    return grid
