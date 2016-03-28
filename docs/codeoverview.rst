.. _codeoverview:

Code Overview
==============

The code provides three main functions:

- Basic polyehdral mesh representation and processing. 
- Construction of Mimetic Finite Difference matrices. 
- Solution to sub-sufrace flow equations (single-phase flow, two-phase flow and the transport equation). 

These three functions are organized into three modules: :ref:`mesh <meshmodule>`,  :ref:`mfd <mfdmodule>` and 
:ref:`models <modelmodule>` (geometry, discretization and physics respectively). 


.. image:: mesh_mfd.svg
   :align: center


The :ref:`mesh <meshmodule>` module contains :ref:`Mesh class <meshclass>`, which is the base class for
representing the geometry of cells. 
In addition, a :ref:`Mesh <meshclass>` instance maintains information about boundary 
faces and some physical properties of the problem.  
Typically, the user will not interact with the :ref:`Mesh <meshclass>` class directly, but rather use subclasses and helper tools 
that make the task much easier. The user can also import files generated using standard mesh generation programs:

.. image:: mesh_sources.svg
   :align: center

The :ref:`mfd <mfdmodule>` module contains the :ref:`MFD <mfdclass>`, which stands for Mimetic Finite Differences. 
:ref:`MFD <mfdclass>` takes a :ref:`Mesh class <meshclass>` instance (or child)
and constructs the discretization matrices.
The :ref:`MFD <mfdclass>` class can set Dirichlet (pressure), 
Neumann (flux) boundary conditions, and forcing terms.

For solving more complex physical problems, the module  :ref:`models <modelmodule>` contains classes that 
solve more complex systems such as single-phase and two-phase flow equations. 

.. image:: models.svg
   :align: center

The :ref:`MFD <mfdclass>` class serves as a critical 
separation between the mesh and model. It allows a model developer 
to solve the diffusion equation without having to directly deal with the geometry.

 
