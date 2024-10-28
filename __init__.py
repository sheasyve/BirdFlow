import bpy
import bmesh  # type: ignore
from mathutils import Vector  # type: ignore
from mathutils.bvhtree import BVHTree  # type: ignore
import numpy as np # type: ignore
from .grid import MACGrid
from .cmain import *

COEFFICIENT_OF_FRICTION = 0.003
EPSILON = 1e-4
PARTICLE_COLOR = [1.0, 1.0, 1.0, 0.5]

bl_info = {
    "name": "Wind Simulator",
    "author": "Shea Syverson",
    "version": (1, 0),
    "blender": (4, 2, 1),
    "location": "View3D > Sidebar > Wind",
    "description": "Wind collision simulation using particles",
    "category": "Object",
}

class WindSim(bpy.types.Operator):
    bl_idname = "object.wind_sim_operator"
    bl_label = "Wind Simulation Operator"
    bl_description = "Simulate wind"

    def get_bvh_tree(self, obj):
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.transform(obj.matrix_world)
        bm.normal_update()
        bvh_tree = BVHTree.FromBMesh(bm)
        bm.free()
        return bvh_tree
    
    def execute(self, context):
        try:
            scene = context.scene
            obj = context.active_object
            if obj and obj.type == 'MESH':
                num_frames = scene.wind_simulation_num_frames
                size = scene.wind_simulation_grid_size
                grid_size = (size, size, size)
                wind_speed_x = scene.wind_simulation_wind_speed_x
                cell_size = scene.wind_simulation_particle_spread
                wind_acceleration_x = scene.wind_simulation_wind_acceleration_x
                damping_factor = scene.wind_simulation_damping_factor
                particle_spread = cell_size
                num_particles = int(size * size * size)
                particle_positions = np.random.rand(num_particles, 3) * particle_spread
                bpy.context.scene.frame_start = 1
                bpy.context.scene.frame_end = num_frames
                wind_speed = np.array([wind_speed_x, 0.0, 0.0], dtype=np.float64)
                wind_acceleration = np.array([wind_acceleration_x, 0.0, 0.0], dtype=np.float64)
                bvh = self.get_bvh_tree(obj)
                self.run_simulation(MACGrid(grid_size, cell_size), 
                    bvh, num_frames,  
                    self.make_particles(self.create_particle_collection(), particle_positions),
                    wind_speed, wind_acceleration,
                    damping_factor, cell_size)
                self.report({'INFO'}, "Simulation finished.")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, "Select a mesh.")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Simulation failed: {e}")
            print(f"Error during simulation: {e}")
            return {'CANCELLED'}

    def create_mesh(self):
        mesh = bpy.data.meshes.get("ParticleMesh")
        if mesh is None:
            mesh = bpy.data.meshes.new("ParticleMesh")
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=8, radius=.025)
            bm.to_mesh(mesh)
            bm.free()
            material = bpy.data.materials.get("ParticleMaterial")
            if material is None:
                material = bpy.data.materials.new(name="ParticleMaterial")
                material.diffuse_color = PARTICLE_COLOR
            if len(mesh.materials) == 0:
                mesh.materials.append(material)
        return mesh

    def create_particle_collection(self):
        pc = bpy.data.collections.get("WindParticles")
        if not pc:
            pc = bpy.data.collections.new("WindParticles")
            bpy.context.scene.collection.children.link(pc)
        return pc

    def make_particles(self, particle_collection, positions):
        mesh = self.create_mesh()
        particle_objects = []
        for i, pos in enumerate(positions):
            particle = bpy.data.objects.new(f"Particle_{i}", mesh)
            particle.location = pos
            particle_collection.objects.link(particle)
            particle_objects.append(particle)
        return particle_objects

    def run_simulation(self, grid, bvh_tree, num_frames,
                       particle_objects, wind_speed, wind_acceleration, 
                       damping_factor, cell_size):
        self.report({'INFO'}, "Running Simulation.")
        dt = 1.
        grid.get_mask(bvh_tree)
        initialize_velocity(grid,.01)
        for frame in range(1, num_frames + 1):
            bpy.context.scene.frame_set(frame)
            cy_simulate(grid, wind_speed, dt, bvh_tree, wind_speed, 
                        wind_acceleration, damping_factor, cell_size, COEFFICIENT_OF_FRICTION) # type: ignore
            grid.update_particle_positions(particle_objects, frame)

class WindSimPanel(bpy.types.Panel):
    bl_label = "Wind Simulator"
    bl_idname = "wind_simulator_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Wind'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.label(text="Simulator Settings:")
        layout.prop(scene, "wind_simulation_grid_size")
        layout.prop(scene, "wind_simulation_wind_speed_x")
        layout.prop(scene, "wind_simulation_wind_acceleration_x")
        layout.prop(scene, "wind_simulation_damping_factor") 
        layout.prop(scene, "wind_simulation_particle_spread")
        layout.prop(scene, "wind_simulation_num_frames")
        layout.operator("object.wind_sim_operator", text="Run Simulation")

def register():
    bpy.utils.register_class(WindSim)
    bpy.utils.register_class(WindSimPanel)
    bpy.types.Scene.wind_simulation_grid_size = bpy.props.IntProperty(
        name="Grid Size",
        description="Grid size",
        default=5,
        min=2,
        max=50
    )
    bpy.types.Scene.wind_simulation_wind_speed_x = bpy.props.FloatProperty(
        name="Wind Speed X",
        description="Speed of wind in the X direction",
        default=.5,
        min=0.0,
        max=100.0
    )
    bpy.types.Scene.wind_simulation_wind_acceleration_x = bpy.props.FloatProperty(
        name="Wind Acceleration X",
        description="Acceleration of wind in the X direction",
        default=0.3,
        min=0.0,
        max=100.0
    )
    bpy.types.Scene.wind_simulation_damping_factor = bpy.props.FloatProperty(
        name="Wind Damping",
        description="Damping factor for wind.",
        default=0.8,
        min=0.0,
        max=1.0
    )
    bpy.types.Scene.wind_simulation_particle_spread = bpy.props.FloatProperty(
        name="Cell Size",
        description="Distance between particles when they are created",
        default=0.5,
        min=0.001,
        max=5.0
    )
    bpy.types.Scene.wind_simulation_num_frames = bpy.props.IntProperty(
        name="Number of Frames",
        description="Total number of frames for the simulation",
        default=100,
        min=1,
        max=1000
    )

def unregister():
    bpy.utils.unregister_class(WindSim)
    bpy.utils.unregister_class(WindSimPanel)
    del bpy.types.Scene.wind_simulation_grid_size
    del bpy.types.Scene.wind_simulation_wind_speed_x
    del bpy.types.Scene.wind_simulation_wind_acceleration_x
    del bpy.types.Scene.wind_simulation_particle_spread
    del bpy.types.Scene.wind_simulation_num_frames
    del bpy.types.Scene.wind_simulation_damping_factor  

if __name__ == "__main__":
    register()
