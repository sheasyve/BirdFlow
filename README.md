# <a href = "https://sheasyve.dev/projects/birdflow.html"> BirdFlow (Blender-Extension) </a>

### Overview

Birdflow is a tool for generating wind simulations in Blender, built with Python and Cython. The extension can be used to generate realistic looking particle based visualizations of air on an object. Realistic air movement beyond naive collision can be observed, as particles avoid high pressure areas and gravitate towards low pressure areas. Collisions are convincing with dynamic velocity damping and redirection, with tangential friction being applied.

https://github.com/user-attachments/assets/8cf4303a-53f4-46a8-b993-bcdad8c6f65f

### Tools, OS, and Libraries

#### OS (Cross Platform not yet working.)

- **Linux**: pop-os 6.9.3-76060903-generic, x86_64 x86_64 GNU/Linux

#### Programs

- **Blender**: 4.21

#### Code

- **Languages**: Python 3.11, Cython 3.0.11

#### Libraries

- **bpy**: Blender API
- **numpy**: Number processing
- **wheels**: Application packaging and distribution
- **cython**: Python to C bridge
- **scipy**: Algorithms and data structures

### Compilation

First install blender for linux from the blender site.

Install the above libraries with pip upon the blenders python installation and possibly system python if needed. Blender's python can be used in the terminal, in the scripting tab of the program.
For me, blender's python was located here.
/home/ssyverson/Documents/blender-4.2.1-linux-x64/4.2/python/bin/python3.11

Modify setup.py so that the include lines point to **your** blenders python, such as /home/ssyverson/Documents/blender-4.2.1-linux-x64/4.2/python/bin/python3.11

Then build the application with Blenders python as below.
The second line is optional, and only for distribution.

```bash
/home/ssyverson/Documents/blender-4.2.1-linux-x64/4.2/python/bin/python3.11 setup.py build_ext --inplace
/home/ssyverson/Documents/blender-4.2.1-linux-x64/4.2/python/bin/python3.11 setup.py bdist_wheel
```

Finally, create simlinks from **each** of the generated .so files to blenders applications folder **Important**. It is a good idea to put them in a subfolder. 
For example, mine are placed here /home/ssyverson/.config/blender/4.2/scripts/addons/cmain_addon
I used these commands.

```bash
ln -s /path/to/grid.cpython-311-x86_64-linux-gnu.so /home/ssyverson/.config/blender/4.2/scripts/addons/cmain_addon/grid.cpython-311-x86_64-linux-gnu.so
ln -s /path/to/cmain.cpython-311-x86_64-linux-gnu.so /home/ssyverson/.config/blender/4.2/scripts/addons/cmain_addon/cmain.cpython-311-x86_64-linux-gnu.so
```

Blender should now see the extension.

### Modeling Strategy

- **Framework**: Eulerian fluid grid based on Navier-Stokes.
- **Integration**: A mix of forward euler and runge-kugga 3 stage integration.
- **Compressibility**: Incompressible flow model. This should be accurate for simulations below Mach 0.3.
- **Boundary Conditions**: No-slip boundary conditions for velocity, Neumann boundary conditions for pressure.
- **Grid Resolution**: Uniform grid with adjustable resolution.
- **Pressure Solver**: Conjugate Gradient Solver from scipy with solid mask built from blender object.
- **Stability and CFL Condition**: Courant-Friedrichs-Lewy condition applied based on maximum velocity in simulation.
- **Numerical Diffusion**: RK3 advection.
- **Visuals**: Wind particle system. Pressure and velocity to particle color coming soon. Vector arrow and volumetric renderings may come.
- **Wind Movement**: Particles independent and influenced by fluid grid, lagrangian particle-object collision.

### Technical Breakdown

The engine uses a Eulerian grid with a Conjugate Gradient solver to simulate incompressible flow by solving the Navier-Stokes equations. Visuals are rendered using particles, for a simple way to display the behavior of the airflow. This involved using Lagrangian methods for particle collisions and movement, making the simulation a hybrid approach having both Eulerian and Lagrangian aspects.

The particles are first advected with a Runge-Kutta 3rd order method, and then pressure in the simulation is calculated using the conjugate gradient solver from SciPy. Pressure changes are distributed through the grid, and particle velocities are updated accordingly after checking for collisions.

## Bibliography

1. Drazin, P. G., & Riley, N. (2006). *The Navier-Stokes equations: A classification of flows and exact solutions* (Vol. 13). Cambridge University Press. [https://books.google.ca/books?hl=en&lr=&id=9SHzrFhVO30C&oi=fnd&pg=PR9&dq=navier-stokes+solutions&ots=buIuaVJAGl&sig=Rb0-MXnH4fKoYrNmVeSFd-8s5OM#v=onepage&q=navier-stokes%20solutions&f=false](https://books.google.ca/books?hl=en&lr=&id=9SHzrFhVO30C&oi=fnd&pg=PR9&dq=navier-stokes+solutions&ots=buIuaVJAGl&sig=Rb0-MXnH4fKoYrNmVeSFd-8s5OM#v=onepage&q=navier-stokes%20solutions&f=false)

2. Braley, C., & Sandu, A. (2020). *Fluid simulation for computer graphics: A tutorial in grid-based and particle-based methods*. Virginia Tech. [https://cg.informatik.uni-freiburg.de/intern/seminar/gridFluids_fluid-EulerParticle.pdf](https://cg.informatik.uni-freiburg.de/intern/seminar/gridFluids_fluid-EulerParticle.pdf)

3. Genevaux, O., Habibi, A., & Dischler, J. M. (2003). *Simulating fluid-solid interaction*. Graphics Interface. [https://graphicsinterface.org/proceedings/gi2003/gi2003-4/](https://graphicsinterface.org/proceedings/gi2003/gi2003-4/)

4. Saad, T. (n.d.). *Turbulence modeling for beginners*. University of Tennessee Space Institute. [https://www.cfd-online.com/W/images/3/31/Turbulence_Modeling_For_Beginners.pdf](https://www.cfd-online.com/W/images/3/31/Turbulence_Modeling_For_Beginners.pdf)

5. Tompson, J., Schlachter, K., Sprechmann, P., & Perlin, K. (2017). *Accelerating Eulerian fluid simulation with convolutional networks*. *Proceedings of the 34th International Conference on Machine Learning*, *Proceedings of Machine Learning Research*, 70, 3424-3433. [https://proceedings.mlr.press/v70/tompson17a.html](https://proceedings.mlr.press/v70/tompson17a.html)

6. He, X., Gao, L., & Wang, Z. (2020). *A hybrid Lagrangian/Eulerian collocated velocity advection and projection method for fluid simulation*. *Computer Graphics Forum*, 39(2), 539-549. [https://doi.org/10.1111/cgf.14096](https://doi.org/10.1111/cgf.14096)
