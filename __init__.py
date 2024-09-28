import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np
from . import cmain  # Import your Cython module

bl_info = {
    "name": "Wind Simulation",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (4, 2, 1),  
    "location": "View3D > Sidebar > Wind",
    "description": "Simulates wind interacting with a stationary object",
    "category": "Object",
}

class WindSimulationOperator(bpy.types.Operator):
    bl_idname = "object.wind_simulation_operator"
    bl_label = "Wind Simulation Operator"
    bl_description = "Simulate wind interacting with a stationary object"

    def execute(self, context):
        obj = context.active_object

        if obj and obj.type == 'MESH':
            num_particles = 500 
            dt = 0.1
            num_frames = 100

            # Create a BVH tree for the object
            mesh = obj.data
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.transform(obj.matrix_world)
            bm.normal_update()

            bvh_tree = BVHTree.FromBMesh(bm)
            bm.free()


            positions = np.random.rand(num_particles, 3) * np.array([10, 10, 10]) - np.array([5, 5, 5])
            velocities = np.zeros((num_particles, 3), dtype=np.float64)

            particle_collection = bpy.data.collections.get("WindParticles")
            if not particle_collection:
                particle_collection = bpy.data.collections.new("WindParticles")
                bpy.context.scene.collection.children.link(particle_collection)

            # Create particle objects using empties for better performance
            particle_objects = []
            for i in range(num_particles):
                particle = bpy.data.objects.new(f"Particle_{i}", None)
                particle.empty_display_type = 'SPHERE'
                particle.empty_display_size = 0.05
                particle.location = positions[i]
                particle_collection.objects.link(particle)
                particle_objects.append(particle)

            # Set up animation frames
            bpy.context.scene.frame_start = 1
            bpy.context.scene.frame_end = num_frames

            # Run simulation and animate particles
            for frame in range(1, num_frames + 1):
                bpy.context.scene.frame_set(frame)

                # Run simulation step without collision detection
                positions, velocities = cmain.simulate(positions, velocities, dt)

                # Perform collision detection in Python
                for i in range(num_particles):
                    origin = positions[i] - velocities[i] * dt
                    velocity_norm = np.linalg.norm(velocities[i])

                    if velocity_norm == 0:
                        continue  # Skip if velocity is zero

                    direction = Vector(velocities[i] / velocity_norm)

                    # Perform ray cast
                    location, normal, index, distance = bvh_tree.ray_cast(
                        Vector(origin), direction, velocity_norm * dt
                    )

                    if location is not None and distance <= velocity_norm * dt:
                        # Adjust position and velocity upon collision
                        positions[i] = np.array(location)
                        normal_np = np.array(normal)
                        velocities[i] = velocities[i] - 2 * np.dot(velocities[i], normal_np) * normal_np

                # Update particle positions
                for i, particle in enumerate(particle_objects):
                    particle.location = positions[i]
                    particle.keyframe_insert(data_path="location", frame=frame)

            self.report({'INFO'}, "Wind simulation completed successfully.")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Please select a mesh object.")
            return {'CANCELLED'}

class WindSimulationPanel(bpy.types.Panel):
    bl_label = "Wind Simulation"
    bl_idname = "OBJECT_PT_wind_simulation_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Wind'

    def draw(self, context):
        layout = self.layout
        layout.operator("object.wind_simulation_operator")

def register():
    bpy.utils.register_class(WindSimulationOperator)
    bpy.utils.register_class(WindSimulationPanel)

def unregister():
    bpy.utils.unregister_class(WindSimulationOperator)
    bpy.utils.unregister_class(WindSimulationPanel)

if __name__ == "__main__":
    register()
