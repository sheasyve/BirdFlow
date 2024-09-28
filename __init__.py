import bpy
import numpy as np
from . import cmain

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
            object_vertices = np.array([obj.matrix_world @ v.co for v in obj.data.vertices], dtype=np.float64)

            positions = np.random.rand(num_particles, 3) * np.array([10, 10, 10]) - np.array([5, 5, 5])
            velocities = np.zeros((num_particles, 3), dtype=np.float64)

            particle_collection = bpy.data.collections.get("WindParticles")
            if not particle_collection:
                particle_collection = bpy.data.collections.new("WindParticles")
                bpy.context.scene.collection.children.link(particle_collection)

            particle_objects = []
            for i in range(num_particles):
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=positions[i])
                particle = bpy.context.active_object
                particle.name = f"Particle_{i}"
                particle.hide_render = True 
                particle_objects.append(particle)
                particle_collection.objects.link(particle)
                collections_to_unlink = [col for col in particle.users_collection if col != particle_collection]
                for col in collections_to_unlink:
                    col.objects.unlink(particle)

            bpy.context.scene.frame_start = 1
            bpy.context.scene.frame_end = num_frames
            for frame in range(1, num_frames + 1):
                bpy.context.scene.frame_set(frame)
                positions, velocities = cmain.simulate(positions, velocities, object_vertices, dt)
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
