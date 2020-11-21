# fringes
Python >= 3.7 package for processing fringe patterns.

Fringes patterns are a kind of special images obtained, for example, from optical interferometry. These images have
a cosine model and can be seeing as two-dimensional cosine signals. When processing fringe patterns, the objective is 
obtain the modulating phase.

The idea of this package is to offer methods and operations to process fringe pattern images.

## Modules

- **psi** has methods and utilities used to demodulate a sequence of phase-shifting fringe patterns. 
- **unwrap** Methodsfor unwrapping 2D phase maps.

## For developers

Each time you modify a cython *pyx* file you need to compile using *setup.py* script. To do this use the following:

    $ python setup.py build_ext --inplace
    
