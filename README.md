# Custom Aerodynamic Simulator Physics Engine (Blender-Extension)

### Proof of concept stage 
<img src="https://github.com/user-attachments/assets/50343c49-76ed-40be-afd1-28e302620c59" alt="0019" width="600"/>
<img src="https://github.com/user-attachments/assets/951b3b26-a395-4636-8b12-4a4dee7ed8f4" alt="0028" width="600"/>
<img src="https://github.com/user-attachments/assets/1deba812-3a6a-4a65-a236-6459627b2c1c" alt="0044" width="600"/>
<img src="https://github.com/user-attachments/assets/f01973f1-6573-463f-b63b-d913fe3cb482" alt="image"/>

My goal is to build a custom aerodynamic simulation framework in Blender, distributed as an open-source extension through the Blender Extensions Platform. My extension will enable users to press a button to have a somewhat realistic aerodynamic simulation animation automatically generated with their current Blender scene.

While the extension will leverage Blender for rendering systems, this will be a custom animation system, as all of the physics calculations for airflow movement, keyframing, and visualization through object creation will be done within my code. I wanted to use blender because it is an amazing free tool that I enjoy using, with robust systems in place.

The user will create their scene in Blender, adjust some parameters in the extension, and click a button to perform the simulation. he extension will use a custom-built physics engine to perform a grid-based Eulerian fluid simulation for airflow over the scene, given the input parameters, which is then automatically animated and keyframed. Currently, my demo is loosely working with very unrealistic physics using a Lagrangian particle simulation.

### Modeling Strategy

- **Framework**: Eulerian fluid grid based on Navier-Stokes, with Lagrangian particle system as a backup.
- **Integration**: Implicit integration using a Backward Euler method. My current demo is using explicit integration.
- **Compressibility**: Incompressible flow model. This should be accurate for simulations below Mach 0.3.
- **Turbulence Model**: k-ε turbulence model.
- **Boundary Conditions**: No-slip boundary condition.
- **Grid Resolution**: Uniform grid with adjustable resolution.
- **Pressure Solver**: Conjugate Gradient Solver.
- **Stability and CFL Condition**: Courant-Friedrichs-Lewy condition.
- **Numerical Diffusion**: MacCormack advection scheme.
- **Visuals**: Particle to grid, vector arrow to grid, and volumetric grid renderings.

### Evaluation Strategy

#### Accuracy

- Comparing the resulting grid to expected calculations.
- Compare the simulation visually to real examples or other simulations.

#### Performance

- Time to generate the animation, or frames per second if real time interaction is implemented.
- Performance should be measured relative to input parameters, like grid density, number of frames and particle count.

#### Plan B: The project can be reduced in scope in the following ways dynamically

- Implement one visualization method.
- Implement less techniques for realistic air flow.
- Continue to use built-in ray casting.
- Abandon real-time interaction.

### Backup Projects

1. **Easier Aerodynamic Engine**: I already have a working prototype using lagrangian methods, I could just improve that.

2. **Liquid simulation**: Liquid simulation has more resources, it would be easier to do liquids.

## 2. Related Work

- Equations needed for Eulerian fluid simulation in many forms [1].

- Detailed instructions for creating a grid-based fluid simulation [2].

- Equations for Navier-Stokes-based fluid simulation and equations for fluid-solid interaction [3].

- Five different models of increasing complexity for turbulence modeling [4].

- Enhancing realism and efficiency using machine learning [5].

- A method for fluid simulation using a hybrid Lagrangian and Eulerian method [6].

(APA citation not used as it was not concise enough.)

## 3. Plan

Distribute the Blender (4.2.1) extension by building it using the Wheels packager (self-contained, cross-platform, easily installed).

### Compilation

```bash
python setup.py build_ext --inplace
python setup.py bdist_wheel
```

### Tools, OS, and Libraries

#### OS

- **Linux**: pop-os 6.9.3-76060903-generic, x86_64 x86_64 GNU/Linux

#### Programs

- **Blender**: 4.21

#### Code

- **Languages**: Python 3.11, Cython 3.0.11, and potentially CUDA 12.6.1

#### Libraries

- **bpy**: Blender API
- **numpy**: Number processing
- **eigen**: Linear algebra
- **wheels**: Application packaging and distribution
- **cython extension**: Python to C bridge
- **scipy**: Algorithms and data structures
- **pyAMG**: Algebraic Multigrid Solver
- **SAMRAI**: AMR Library
- **matplotlib**: Pre-visualization, debugging

#### Textures, Models, Datasets

- The application should require no textures, models, or datasets beyond installing blender.

### Aerodynamic Simulation Project Timeline
![Aerodynamic Simulation Project Timeline](./image.png){ width=50% }

### Potential Difficulties

- Learning the blender API as I go.
- Balancing realistic physics with real-time rendering.
- Learning every algorithm required for this project thoroughly.
- Over-focusing on optimization.

## 4. Bibliography

1. Drazin, P. G., & Riley, N. (2006). *The Navier-Stokes equations: A classification of flows and exact solutions* (Vol. 13). Cambridge University Press. [https://books.google.ca/books?hl=en&lr=&id=9SHzrFhVO30C&oi=fnd&pg=PR9&dq=navier-stokes+solutions&ots=buIuaVJAGl&sig=Rb0-MXnH4fKoYrNmVeSFd-8s5OM#v=onepage&q=navier-stokes%20solutions&f=false](https://books.google.ca/books?hl=en&lr=&id=9SHzrFhVO30C&oi=fnd&pg=PR9&dq=navier-stokes+solutions&ots=buIuaVJAGl&sig=Rb0-MXnH4fKoYrNmVeSFd-8s5OM#v=onepage&q=navier-stokes%20solutions&f=false)

2. Braley, C., & Sandu, A. (2020). *Fluid simulation for computer graphics: A tutorial in grid-based and particle-based methods*. Virginia Tech. [https://cg.informatik.uni-freiburg.de/intern/seminar/gridFluids_fluid-EulerParticle.pdf](https://cg.informatik.uni-freiburg.de/intern/seminar/gridFluids_fluid-EulerParticle.pdf)

3. Genevaux, O., Habibi, A., & Dischler, J. M. (2003). *Simulating fluid-solid interaction*. Graphics Interface. [https://graphicsinterface.org/proceedings/gi2003/gi2003-4/](https://graphicsinterface.org/proceedings/gi2003/gi2003-4/)

4. Saad, T. (n.d.). *Turbulence modeling for beginners*. University of Tennessee Space Institute. [https://www.cfd-online.com/W/images/3/31/Turbulence_Modeling_For_Beginners.pdf](https://www.cfd-online.com/W/images/3/31/Turbulence_Modeling_For_Beginners.pdf)

5. Tompson, J., Schlachter, K., Sprechmann, P., & Perlin, K. (2017). *Accelerating Eulerian fluid simulation with convolutional networks*. *Proceedings of the 34th International Conference on Machine Learning*, *Proceedings of Machine Learning Research*, 70, 3424-3433. [https://proceedings.mlr.press/v70/tompson17a.html](https://proceedings.mlr.press/v70/tompson17a.html)

6. He, X., Gao, L., & Wang, Z. (2020). *A hybrid Lagrangian/Eulerian collocated velocity advection and projection method for fluid simulation*. *Computer Graphics Forum*, 39(2), 539-549. [https://doi.org/10.1111/cgf.14096](https://doi.org/10.1111/cgf.14096)
