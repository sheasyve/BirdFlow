# Blender-Extension
## Compilation for now
/home/ssyverson/Documents/blender-4.2.1-linux-x64/4.2/python/bin/python3.11 setup.py build_ext --inplace

Should move the extension to the blender folder eventually

## Animation Project starting phase

# Notes 
python setup.py build_ext --inplace
python setup.py bdist_wheel

This command will generate a .whl file in the dist/ directory. You can then distribute this wheel, and it can be installed via pip using:

pip install dist/your_wheel_file.whl

you can move your wheel file there after building it:

bash
Copy code
mv dist/*.whl wheels/


### In blender:
import importlib
import blender_extension  
importlib.reload(blender_extension)\
bpy.ops.script.reload()


#### In init.py
import bpy
import importlib
if "my_extension" in locals():
    importlib.reload(my_extension)
else:
    from . import my_extension


2. Point Blender to a Custom Add-on Directory:
Blender allows you to set custom directories for add-ons. Here’s how to configure it:

Open Blender Preferences:
In Blender, go to Edit > Preferences (or Blender > Preferences on macOS).

Set a Custom Add-on Path:

In the Preferences window, click on the File Paths tab.
Find the Scripts section and click the folder icon next to it.
Navigate to the folder where your add-on is located (your custom directory where you have your __init__.py and the compiled Cython .so file).
Save Preferences:

Once you've set the custom add-on folder path, click the Save Preferences button in the lower-left corner of the Preferences window.
Install and Enable the Add-on:

Go to the Add-ons tab, click Install, and select the __init__.py file from your custom folder.
After installing, make sure to enable the add-on by checking the box next to it.
3. Check Add-on Path via Python Script:
You can also confirm where Blender is looking for add-ons by running this Python script inside Blender's scripting tab:

python
Copy code
import bpy
print(bpy.utils.script_paths("addons"))
This will print out the directories Blender is currently using for add-ons.

4. Rebuilding Workflow:
Once you've set Blender to look at the correct folder:

Continue running the python setup.py build_ext --inplace command from that same directory.
Blender will pick up the updated .so or .pyd file from the custom folder after you either reload the add-on or use the auto-reload script.
