import bpy

bl_info = {
    "name": "Test Extension",
    "author": "Shea",
    "blender": (4, 2, 1),
    "category": "Object"
}

def test_operator(self, context):
    self.layout.label(text="Hello World")

def menu_func(self, context):
    self.layout.operator("object.test_operator")

class Extension(bpy.types.Operator):
    bl_idname = "object.test_operator"
    bl_label = "Test Operator"
    
    def execute(self, context):
        self.report({'INFO'}, "Hello World")
        return {'FINISHED'}

class CustomPanel(bpy.types.Panel):
    bl_label = "Custom Panel"
    bl_idname = "OBJECT_PT_custom_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'
    
    def draw(self, context):
        layout = self.layout
        obj = context.object
        layout.label(text="Hello World")
        layout.operator("object.test_operator")
        layout.prop(obj, "my_custom_property")

bpy.types.Object.my_custom_property = bpy.props.StringProperty(
    name="My Custom Property",
    description="A custom property for objects",
    default="Default Value"
)

def register():
    bpy.utils.register_class(Extension)
    bpy.utils.register_class(CustomPanel)
    bpy.types.VIEW3D_MT_object.append(menu_func)

def unregister():
    bpy.utils.unregister_class(Extension)
    bpy.utils.unregister_class(CustomPanel)
    bpy.types.VIEW3D_MT_object.remove(menu_func)
    del bpy.types.Object.my_custom_property

def main():
    register()

if __name__ == "__main__":
    main()