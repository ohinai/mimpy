
Introduction
============


Mimpy solves equations like the diffusion problem:

.. math::
     \begin{align}
     - \nabla \cdot K \nabla p = f
     \end{align}

using the Mimetic Finite Difference (MFD) method.
MFD, like many other discretization methods, 
requires the construction of a mesh. However, unlike most methods that require a specific 
kind of mesh, like triangles or rectangles, the MFD method can use general polyhedral meshes. This means 
that it can naturally solve using traditional meshes like tetrahedra, hexahedra and Voronoi:

.. image:: three_solutions.png
 
as well as hybrids of these meshes and many more.

The code has three main parts:
 
- **Mesh**: A representation of general polyhedral meshes and some basic operations on them. 
- **MFD** : Produces the MFD matrices used for solving certain problems over these meshes. 
- **models**: A module that contains classes for solving physical problems such as single-phase flow, the transport equation and two-phase flow.  










