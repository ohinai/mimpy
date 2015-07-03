.. _codeoverview:

Code Overview
==============

Mimpy is designed to be a simple library, with a transparent set of data structures and 
methods.
The code provides three main functions:

- Basic polyehdral mesh representation and processing. 
- Construction of Mimetic Finite Difference matrices. 
- Solution to sub-sufrace flow equations (single-phase flow, two-phase flow and the transport equation). 

These three functions are organized into three modules: **mesh**, **mfd** and **models**:


.. image:: mesh_mfd.svg
   :align: center


The **mesh** module contains the **Mesh** class which 
represents the geometry of the mesh that will be used for computation. 
In addition, a **Mesh** instance maintains information about boundary 
faces and some physical properties of the problem.  
Typically, the user will not interact with the **Mesh** class directly, but rather use subclasses and helper tools 
that make the task much easier. The user can also import files generated using standard mesh generation programs:

.. image:: mesh_sources.svg
   :align: center

The module **mfd** contains the **MFD** class, and stands for Mimetic Finite Difference. 
**MFD** takes a **Mesh** instance 
and constructs the discretization matrices.
The **MFD** class can set Dirichlet (pressure), Neumann (flux) boundary conditions, and forcing terms. 

For solving more complex physical problems, the module  **models** contains classes that 
solve more complex systems such as single-phase and two-phase flow equations. 

.. image:: models.svg
   :align: center

 
