from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "cmain", 
        sources=["cmain.pyx"], 
        include_dirs=[np.get_include()],  
        libraries=["stdc++"],            
    )
]

setup(
    name="blender_extension",
    version="1.0",
    description="A Blender extension using Cython",
    ext_modules=cythonize(extensions),   
    setup_requires=['numpy', 'cython', 'wheel'],  
    install_requires=['numpy', 'scipy'], 
)