
Introduction
============


Mimpy solves equations 
like the diffusion problem:

.. math::
     \begin{align}
     - \nabla \cdot K \nabla p = f
     \end{align}

by finding discrete operators (matrices) that approximate the gradient 
and divergence:

.. math::
     \begin{align}
     \mathcal{DIV} &\approx \nabla \cdot\\
     \mathcal{G} &\approx  K \nabla\\      
     \end{align}

Resulting in the approximation:

.. math::
     -\mathcal{DIV} \mathcal{G} p_h = f^I


The process of taking the continuous problem and finding a finite dimensional 
approximation is called *discretization*. Mimpy uses a discretization method called
the Mimetic Finite Difference (MFD) method. MFD, like most other methods, 
requires the construction of a mesh. However, unlike most methods that require a specific 
kind of mesh, like triangles or rectangles, the MFD method can use general polyhedra. This means 
that it can solve using traditional meshes:

.. image:: three_solutions.png

In addition to meshes that are not so traditional, and any combination in between.
The code has three main parts:
 
- **Mesh**: Represents the geometry of the mesh. 
- **MFD** : Takes a mesh and generates the discrete operators. 
- **models**: A module that contains classes for solving physical problems such as single-phase flow, the transport equation and two-phase flow.  










