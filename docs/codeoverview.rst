.. _codeoverview:

Code Overview
==============

Mimpy is designed to be a very simple library, with a transparent set of data structures and 
methods.
There are two main classes, **Mesh** and **MFD**: 

.. image:: mesh_mfd.svg
   :align: center


**Mesh** represents the geometry of the mesh that will be used for computation. 
In addition, a **Mesh** instance maintains information about boundary faces and some physical properties of the problem.  
Typically, the user will not interact with the **Mesh** class directly, but rather use subclasses and helper tools 
that make the task much easier. The user can also import files generated using standard mesh generation programs:

.. image:: mesh_sources.svg
   :align: center

The second class is called **MFD**, and stands for Mimetic Finite Difference. **MFD** takes a **Mesh** instance 
and constructs the discretization matrices. 
The **MFD** class can set Dirichlet (pressure), Neumann (flux) boundary conditions, and forcing terms. 

The **Mesh** and **MFD** together are sufficient for solving the diffusion equation. 
For solving more complex physical problems, the module  **models** contains classes that 
solve more complex systems such as single-phase and two-phase flow equations. 

.. image:: models.svg
   :align: center

The three branches of the code, **Mesh**, **MFD** and **models**,  
are orthogonal entities. For example, 
the single-phase model is not aware of the specific 
kind of mesh used, and will work with tetrahedral meshes as well Voronoi meshes. 
 
