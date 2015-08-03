
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
kind of mesh (like tetrahedra or hexahedra), the MFD method accepts general polyhedral elements. This means 
that it can naturally solve using traditional meshes like tetrahedra, hexahedra and Voronoi:

.. image:: three_solutions.png

and many more. This feature is uselful since:

- Users don't always have control over the mesh they use.
- The element flexibility allows for  modeling complex features such as fractures. 
- It allows model developers to debug their models on simple rectangular meshes 
  first, and then move on to more complex geometries.
- Polyhedral elements make for a natural way to resolve local grid refinements. 

The diffusion equation is often referred to a "kernel" problem for many more complex sets of 
equations. An example of such equations are the ones related to porous media flows.

For more information on the software, check out the :ref:`codeoverview`. 

.. _discretization: http://en.wikipedia.org/wiki/Numerical_partial_differential_equations


.. [BLS2005] Brezzi, Franco, Konstantin Lipnikov, and Mikhail Shashkov. "Convergence of the mimetic finite difference method 
     for diffusion problems on polyhedral meshes." SIAM Journal on Numerical Analysis 43.5 (2005): 1872-1896.



