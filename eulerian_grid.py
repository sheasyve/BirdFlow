import bpy
import bmesh  # type: ignore
from mathutils import Vector  # type: ignore
from mathutils.bvhtree import BVHTree  # type: ignore
import numpy as np  # type: ignore
from .cmain import EulerianGrid  # Use the Cython version directly
from . import cmain

# Initialize EulerianGrid using the Cython class
grid_size = (10, 10, 10)  # Example size
cell_size = 1.0  # Example cell size

# Create the EulerianGrid using the Cython class
eulerian_grid = EulerianGrid(grid_size, cell_size)

# Now use Cython-optimized methods
wind = np.array([1.0, 0.0, 0.0], dtype=np.float64)
dt = 0.1

# Simulate
cmain.cy_simulate(eulerian_grid, wind, dt)

# For collision detection:
bvh_tree = None  # Replace with your actual BVHTree object
cmain.cy_collide(eulerian_grid, bvh_tree, dt)
