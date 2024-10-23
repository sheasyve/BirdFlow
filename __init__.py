import bpy
import bmesh  # type: ignore
from mathutils import Vector  # type: ignore
from mathutils.bvhtree import BVHTree  # type: ignore
import numpy as np

from .cmain import MACGrid

COEFFICIENT_OF_FRICTION = 1.0
EPSILON = 1e-4
WIND_DIRECTION = [1.0, 0.0, 0.0]
PARTICLE_COLOR = [1.0, 1.0, 1.0, 0.5]
GRID_SIZE = (5, 5, 5) 
CELL_SIZE = 0.5         

bl_info = {
    "name": "Aerodynamic Simulator",
    "author": "Shea Syverson",
    "version": (1, 0),
    "blender": (4, 2, 1),
    "location": "View3D > Sidebar > Wind",
    "description": "Aerodynamic collision simulation using particles",
    "category": "Object",
}

class WindSim(bpy.types.Operator):
    bl_idname = "object.wind_sim_operator"
    bl_label = "Wind Simulation Operator"
    bl_description = "Simulate wind"

    num_particles = bpy.props.IntProperty(
        name="Number of Particles",
        description="Total number of wind particles",
        default=125,
        min=1,
        max=10000
    )
    particle_spread = bpy.props.FloatProperty(
        name="Particle Spread",
        description="Distance between particles",
        default=0.1,
        min=0.001,
        max=100.0
    )
    num_frames = bpy.props.IntProperty(
        name="Number of Frames",
        description="Total number of frames for the simulation",
        default=100,
        min=1,
        max=1000
    )
    dt = bpy.props.FloatProperty(
        name="Time Step",
        description="Time step for the simulation",
        default=0.1,
        min=0.001,
        max=1.0
    )

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
                num_frames = scene.aerodynamic_simulation_num_frames
                dt = scene.aerodynamic_simulation_dt
                num_particles = scene.aerodynamic_simulation_num_particles
                particle_spread = scene.aerodynamic_simulation_particle_spread 
                particle_positions = np.random.rand(num_particles, 3) * particle_spread
                bpy.context.scene.frame_start = 1
                bpy.context.scene.frame_end = num_frames
                self.run_simulation(
                    MACGrid(GRID_SIZE, CELL_SIZE), 
                    self.get_bvh_tree(obj), num_frames, 
                    dt, self.make_particles(self.create_particle_collection(), 
                    particle_positions))
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
        # Create the particle mesh where the particles will be displayed
        mesh = bpy.data.meshes.get("ParticleMesh")
        if mesh is None:
            mesh = bpy.data.meshes.new("ParticleMesh")
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=8, radius=.025)  # Particle specs, notably radius
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
        # Create or retrieve the particle collection in Blender
        pc = bpy.data.collections.get("WindParticles")
        if not pc:
            pc = bpy.data.collections.new("WindParticles")
            bpy.context.scene.collection.children.link(pc)
        return pc

    def make_particles(self, particle_collection, positions):
        # Initialize the particles
        mesh = self.create_mesh()
        particle_objects = []
        for i, pos in enumerate(positions):  # One particle per position
            particle = bpy.data.objects.new(f"Particle_{i}", mesh)
            particle.location = pos
            particle_collection.objects.link(particle)
            particle_objects.append(particle)
        return particle_objects

    def run_simulation(self, grid, bvh_tree, num_frames, dt, particle_objects):
        wind = np.array(WIND_DIRECTION, dtype=np.float64)
        for frame in range(1, num_frames + 1):
            bpy.context.scene.frame_set(frame)
            grid.simulate(wind, dt, bvh_tree)
            grid.update_particle_positions(particle_objects, frame)

class WindSimPanel(bpy.types.Panel):
    # Create the panel in Blender for the simulator
    bl_label = "Aerodynamic Simulator"
    bl_idname = "aerodynamic_simulator_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Wind'
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.label(text="Simulator Settings:")
        layout.prop(scene, "aerodynamic_simulation_num_particles")
        layout.prop(scene, "aerodynamic_simulation_particle_spread")
        layout.prop(scene, "aerodynamic_simulation_num_frames")
        layout.prop(scene, "aerodynamic_simulation_dt")
        layout.operator("object.wind_sim_operator", text="Run Simulation")

def register():
    # Register the simulator class, panel, and panel options in Blender
    bpy.utils.register_class(WindSim)
    bpy.utils.register_class(WindSimPanel)
    bpy.types.Scene.aerodynamic_simulation_num_particles = bpy.props.IntProperty(
        name="Number of Particles",
        description="Total number of wind particles",
        default=500,
        min=1,
        max=10000
    )
    bpy.types.Scene.aerodynamic_simulation_particle_spread = bpy.props.FloatProperty(
        name="Particle Spread",
        description="Distance between particles",
        default=0.1,
        min=0.001,
        max=100.0
    )
    bpy.types.Scene.aerodynamic_simulation_num_frames = bpy.props.IntProperty(
        name="Number of Frames",
        description="Total number of frames for the simulation",
        default=100,
        min=1,
        max=1000
    )
    bpy.types.Scene.aerodynamic_simulation_dt = bpy.props.FloatProperty(
        name="Time Step",
        description="Time step for the simulation",
        default=0.1,
        min=0.001,
        max=1.0
    )

def unregister():
    bpy.utils.unregister_class(WindSim)
    bpy.utils.unregister_class(WindSimPanel)
    del bpy.types.Scene.aerodynamic_simulation_num_particles
    del bpy.types.Scene.aerodynamic_simulation_particle_spread
    del bpy.types.Scene.aerodynamic_simulation_num_frames
    del bpy.types.Scene.aerodynamic_simulation_dt


if __name__ == "__main__":
    register()
