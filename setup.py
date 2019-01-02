from distutils.core import setup

from Cython.Build import cythonize


cython_modules = [
    "./fringes/scanning/pixel.pyx"
]

setup(
    ext_modules=cythonize(cython_modules)
)