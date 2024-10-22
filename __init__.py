import bpy
import bmesh  # type: ignore
from mathutils import Vector  # type: ignore
from mathutils.bvhtree import BVHTree  # type: ignore
import numpy as np
from . import cmain

COEFFICIENT_OF_FRICTION = 1.0
EPSILON = 1e-4
WIND_DIRECTION = [1.0, 0.0, 0.0]
PARTICLE_COLOR = [1.0, 1.0, 1.0, 0.5]

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
        default=500,
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
                num_particles = scene.aerodynamic_simulation_num_particles
                dt = scene.aerodynamic_simulation_dt
                num_frames = scene.aerodynamic_simulation_num_frames
                spread = scene.aerodynamic_simulation_particle_spread
                p_loc = 10 * spread
                positions = np.random.rand(num_particles, 3) * np.array([p_loc, p_loc, p_loc]) - np.array([p_loc / 2, p_loc / 2, p_loc / 2])
                velocities = np.zeros((num_particles, 3), dtype=np.float64)
                particles = self.make_particles(self.create_particle_collection(), positions)
                bpy.context.scene.frame_start = 1
                bpy.context.scene.frame_end = num_frames
                self.run_simulation(positions, velocities, self.get_bvh_tree(obj), num_frames, dt, particles)
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
        #Create the particle mesh where the particles will be displayed
        mesh = bpy.data.meshes.get("ParticleMesh")
        if mesh is None:
            mesh = bpy.data.meshes.new("ParticleMesh")
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=8, radius=.025)#Particle specs, notably r
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
        #Initialize the particles
        mesh = self.create_mesh()
        particle_objects = []
        for i, pos in enumerate(positions):#One particle per position
            particle = bpy.data.objects.new(f"Particle_{i}", mesh)
            particle.location = pos
            particle_collection.objects.link(particle)
            particle_objects.append(particle)
        return particle_objects

    def redirect(self, positions, velocities,location, normal, i):
        #Reflect the particle off the surface
        normal = np.array(normal)
        normal /= np.linalg.norm(normal)
        v_normal = np.dot(velocities[i], normal) * normal #Deflect the particle
        velocities[i] = velocities[i] - v_normal
        velocities[i] *= COEFFICIENT_OF_FRICTION
        positions[i] = np.array(location) + normal * EPSILON

    def run_simulation(self, positions, velocities, bvh_tree, num_frames, dt, particle_objects):
        wind = np.array(WIND_DIRECTION, dtype=np.float64)
        for frame in range(1, num_frames + 1):
            bpy.context.scene.frame_set(frame)
            positions, velocities = cmain.simulate(positions, velocities, dt, wind)#Call the c++ function for wind simulation
            for i in range(len(positions)):#Check for collisions with ray casting
                origin = positions[i] - velocities[i] * dt
                velocity_norm = np.linalg.norm(velocities[i])
                if velocity_norm == 0:
                    continue
                direction = Vector(velocities[i] / velocity_norm)
                location, normal, index, distance = bvh_tree.ray_cast(Vector(origin), direction, velocity_norm * dt)
                if location is not None and distance <= velocity_norm * dt:#Intersecting with the mesh, redirect the particle
                    self.redirect(positions, velocities, location, normal, i)
            for i, particle in enumerate(particle_objects):#Update the particle locations
                particle.location = positions[i]
                particle.keyframe_insert(data_path="location", frame=frame)
    
    def run_simulation2(self, eulerian_grid, bvh_tree, num_frames, dt, particle_objects):
        wind = np.array(WIND_DIRECTION, dtype=np.float64)
        for frame in range(1, num_frames + 1):
            bpy.context.scene.frame_set(frame)
            eulerian_grid.collide(bvh_tree, dt)
            eulerian_grid.simulate(wind, dt)
            eulerian_grid.update_particle_positions(particle_objects, frame)

class WindSimPanel(bpy.types.Panel):
    #Create the panel in blender for the simulator
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
    #Register the simulator class, panel, and panel options in blender
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
