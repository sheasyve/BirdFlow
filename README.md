# Custom Aerodynamic Simulator Physics Engine (Blender-Extension)
# Animation Project - Proof of concept stage 
![image](https://github.com/user-attachments/assets/8ea0280e-d33f-4749-b64b-15d30882ccf9)
![0019](https://github.com/user-attachments/assets/50343c49-76ed-40be-afd1-28e302620c59)
![0028](https://github.com/user-attachments/assets/951b3b26-a395-4636-8b12-4a4dee7ed8f4)
![0044](https://github.com/user-attachments/assets/1deba812-3a6a-4a65-a236-6459627b2c1c)

# My goal is to build a custom aerodynamic simulation framework in blender, distributed as an extension.
# The particle mesh management and physics simulation will be custom built with python and cython. 
# I am not using the built in blender simulations, rather using blender as a rendering environment for my code.
# For the final deliverable, I hope to implement most of the following.
- ## Advanced aerodynamic visualisation
    - I would like to be able to import any model, particularly cars, and use my extension to quickly generate an accurate and stunning visualisation of wind particle flow.
    - Advanced custom particle physics engine that mimic wind behaviour accurately and efficiently.
    - Full set of parameters like wind speed, direction, and others.
- ## Easily tuneable simulation
    - Allow for different wind simulation paramaters.
          - More realistic wind patterns, wind tunnel setups, storm conditions.
    - Optimise to allow for live interaction.
          - It would be cool to click with the mouse and see wind particles be placed at a certain area, to better understand the aerodynamics. Not sure if this one is achievable but it might be in the viewport.
    - Distribute the extension officially through blender.

### Currently, I have a simple custom wind particle physics engine set up as a blender extension.
### The extension is built in python, cython (c in python for heavy tasks) and uses the bpy library to interact with blender.
### For now, the user can input some parameters like particle amount and density, and a simulation will be automatically generated in blender.
### The physics are very unrealistic right now, but there is still collision detection, sliding and redirection set up (somewhat)

## Compilation for now
```bash
/home/ssyverson/Documents/blender-4.2.1-linux-x64/4.2/python/bin/python3.11 setup.py build_ext --inplace
```
Should move the extension to the blender folder eventually

#### Wheel Notes 
```bash
python setup.py build_ext --inplace
python setup.py bdist_wheel
```
This command will generate a .whl file in the dist/ directory. You can then distribute this wheel, and it can be installed via pip using:
```bash
pip install dist/your_wheel_file.whl
```
you can move your wheel file there after building it:

```bash
mv dist/*.whl wheels/
```


