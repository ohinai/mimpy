=====
mimpy
=====

.. image:: https://api.travis-ci.org/ohinai/mimpy.png?branch=master
    :target: https://travis-ci.org/ohinai/mimpy

Mimpy is a Python based library for solving
the diffusion equation. Mimpy uses the Mimetic Finite Difference 
method, which as the advantage of solving using general polyhedral elements. 

Because diffusion is a "kernel" for many physical phenomena, 
mimpy can also solve problems like the heat equation and multi-phase flow
through porous media. 

A simple example can be found in hexmesh_example_1.py_.
Run:

    .. code-block:: bash
    
        $ python hexmesh_example_1.py 

Code requirements: 

* NumPy 
* SciPy
* Cython (if you want to update the c code). 


