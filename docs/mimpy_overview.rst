Mimpy Overview
==============

The primary function of mimpy is to solve the diffusion equation:

.. math::
    -\nabla \cdot K \nabla p = f

using the Mimetic Finite Difference (MFD) method defined in [B2005A]_ and [B2005B]_. The two main classes in mimpy are **mesh** and **mfd**. 
The **mesh** class is capable of directly representing general polyhedral elements, which naturally includes conventional meshes such 
tetrahedra, hexahedra and Voronoi. The **mfd** class processes the mesh information and produces the linear system of equations associated with it, 
which after solving produces the approximate solution to our diffusion problem. From here, it is possible to use these classes to solve more 
complex flow problems, such as the single-phase and two-phase flow equations. Classes that solve these equations are found under the **models**
module. 


mesh
----

The **mesh** is the fundemental data structure in mimpy, and 








.. [B2005A] Brezzi, Franco, Konstantin Lipnikov, and Mikhail Shashkov. "Convergence of the mimetic finite difference method for diffusion problems on polyhedral meshes." SIAM Journal on Numerical Analysis 43.5 (2005): 1872-1896.

.. [B2005B] Brezzi, Franco, Konstantin Lipnikov, and Valeria Simoncini. "A family of mimetic finite difference methods on polygonal and polyhedral meshes." Mathematical Models and Methods in Applied Sciences 15.10 (2005): 1533-1551.

