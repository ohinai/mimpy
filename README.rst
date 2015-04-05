=====
mimpy
=====

.. image:: https://api.travis-ci.org/ohinai/mimpy.png?branch=master
    :target: https://travis-ci.org/ohinai/mimpy

Mimpy is a Python based library for solving
the diffusion equation:

.. image:: diff_eq.gif

Because diffusion is a "kernel" for many physical phenomena, 
mimpy can also solve problems like the heat equation and multi-phase flow
through porous media. 

Mimpy is based on the Mimetic Finite Difference method, which allows 
it to solve over general polyhedral elements. That includes hexahedra, 
tetrahedra and Voronoi elements:

.. image:: three_solutions.png

The best way to get the code right now is to clone the git repo and run the setup utility:

    .. code-block:: bash
    
        $ git clone https://github.com/ohinai/mimpy.git
        $ python setup.py install 

A simple example can be found in hexmesh_example_1.py_.
Run:

    .. code-block:: bash
    
        $ python hexmesh_example_1.py 

If all goes well, you should get a file named (hexmesh_example_1.vtk). Open the file using 
Paraview and plot "MFDPressure." You should see something like this:

.. image:: hexmesh_solution.png




.. _hexmesh_example_1.py: https://github.com/ohinai/mimpy/blob/master/examples/hexmesh/example1/hexmesh_example_1.py


