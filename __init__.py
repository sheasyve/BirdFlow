import bpy
import numpy as np
from . import cmain  

bl_info = {
    "name": "Test Extension",
    "author": "Shea",
    "blender": (4, 2, 1),  
    "category": "Object"
}

class Extension(bpy.types.Operator):
    bl_idname = "object.test_operator"
    bl_label = "Test Operator"
    def execute(self, context):
        obj = bpy.context.active_object
        if obj and obj.type == 'MESH':
            vertices = np.array([v.co for v in obj.data.vertices], dtype=np.float64)
            print("Mesh Vertices:", vertices)
            velocity = np.zeros((10, 10), dtype=np.float64)
            pressure = np.random.rand(10, 10)
            dt = 0.01
            updated_velocity = cmain.simulate(velocity, pressure, dt)
            print("Updated Velocity Field:", updated_velocity)
            self.report({'INFO'}, "Simulation complete")
        else:
            self.report({'ERROR'}, "No mesh object selected")
        return {'FINISHED'}

class CustomPanel(bpy.types.Panel):
    bl_label = "Aero Panel"
    bl_idname = "OBJECT_PT_custom_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'
    
    def draw(self, context):
        layout = self.layout
        obj = context.object
        layout.label(text="Shea's Custom Panel")
        layout.operator("object.test_operator")
        layout.prop(obj, "my_custom_property")

bpy.types.Object.my_custom_property = bpy.props.StringProperty(
    name="Shea's Custom Property",
    description="A custom property for objects",
    default="Default Value"
)

def register():
    bpy.utils.register_class(Extension)
    bpy.utils.register_class(CustomPanel)

def unregister():
    bpy.utils.unregister_class(Extension)
    bpy.utils.unregister_class(CustomPanel)

def main():
    register()

if __name__ == "__main__":
    main()
