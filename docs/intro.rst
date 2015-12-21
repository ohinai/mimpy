
Introduction
============


Mimpy solves the diffusion problem:

.. math::
     \begin{align}
     - \nabla \cdot K \nabla p = f
     \end{align}

using the Mimetic Finite Difference (MFD) method [BLS2005]_. 
The diffusion equation is often referred to a "kernel" problem for many more complex sets of 
equations. An example of such equations are the ones related to porous media flows.

The MFD method, like many other discretization_ methods, 
requires the construction of a mesh. However, unlike most conventional methods that are restricted to a specific 
kind of element, the MFD method accepts a wide range of general polyhedral elements. This means 
that it can naturally solve for all kinds of conventional meshes such as tetrahedra, hexahedra and Voronoi:

.. image:: three_solutions.png

and many more. This feature is uselful since:

- Users don't always have control over the mesh they use.
- It allows model developers to debug their models on simple rectangular meshes 
  first, and then move on to more complex geometries.
- You can mix and match meshes. For example, solving one part of the domain with 
  rectangles and the other with triangles. 
- Polyhedral elements naturally allow for local grid refinement (adaptivity). 


Mimpy is designed for simple and intuitive use. 
Meshes can be constructed using built-in routines or 
loaded from popular mesh files. Boundary faces can be identified and solved for using a simple boundary marker construct. 


For more information on the software, check out the :ref:`code overview <codeoverview>`. 


For porous media model examples, see  :ref:`gallery of examples <examplegallery>` and 
:ref:`model classes <modelmodule>`. 


.. _discretization: http://en.wikipedia.org/wiki/Numerical_partial_differential_equations


.. [BLS2005] Brezzi, Franco, Konstantin Lipnikov, and Mikhail Shashkov. "Convergence of the mimetic finite difference method 
     for diffusion problems on polyhedral meshes." SIAM Journal on Numerical Analysis 43.5 (2005): 1872-1896.



