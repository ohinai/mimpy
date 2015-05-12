
Introduction
============


Mimpy solves equations like the diffusion problem:

.. math::
     \begin{align}
     - \nabla \cdot K \nabla p = f
     \end{align}

using the Mimetic Finite Difference (MFD) method [BLS2005]_.
MFD, like many other discretization_ methods, 
requires the construction of a mesh. However, unlike most methods that require a specific 
kind of mesh (like tetrahedra or hexahedra), the MFD method can use general polyhedral meshes. This means 
that it can naturally solve using traditional meshes like tetrahedra, hexahedra and Voronoi:

.. image:: three_solutions.png
 
as well as hybrids of these meshes and many more. This feature is very uselful since often times 
the user may not have control over mesh generation. In addition, the mesh flexibility allows  
users to first validate models against simple rectangular grids, and then move on to more complex 
problems.  

Mimpy follows a very simple organization through out, just remember the three "m"s:

- **mesh**: A representation of general polyhedral meshes and some basic operations on them. 
- **matrix** : The linear systems of equations produced from the MFD discretization. 
- **model**: Physical problems such as single-phase flow, the transport equation and two-phase flow.

For more information on the software, check out the :ref:`codeoverview`. 


.. _discretization: http://en.wikipedia.org/wiki/Numerical_partial_differential_equations


.. [BLS2005] Brezzi, Franco, Konstantin Lipnikov, and Mikhail Shashkov. "Convergence of the mimetic finite difference method 
     for diffusion problems on polyhedral meshes." SIAM Journal on Numerical Analysis 43.5 (2005): 1872-1896.



