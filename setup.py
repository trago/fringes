from distutils.core import setup
from Cython.Build import cythonize

cython_modules = [
    "./fringes/scanning/pixel.pyx",
    "./fringes/scanning/floodfill.pyx"
]

compile_directives = {
    'language_level': 2 # does not compiles with level= 3 or 3str
}

ext_modules = cythonize(cython_modules, compiler_directives=compile_directives)

setup(
    name="Fringe analysis for python",
    ext_modules=ext_modules
)
