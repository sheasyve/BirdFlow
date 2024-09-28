from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from Cython.Compiler import Options
Options.language_level = 3

extensions = [
    Extension(
        "cmain",
        sources=["cmain.pyx"],
        include_dirs=[np.get_include()],
        libraries=["stdc++"],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
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
