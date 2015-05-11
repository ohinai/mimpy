

Single-Phase Example
====================


We'll now take a walk-through example for solving the single-phase flow problem:

.. math::
     \begin{align} 
     \phi \frac{\partial \rho}{\partial t} = \nabla \cdot \frac{\rho}{\mu}K \nabla p + f
     \end{align}


We'll first need a few import statements, starting 
with NumPy:

.. code-block:: python

     import numpy as np

Next, we'll need the following from Mimpy:

.. code-block:: python

     import mimpy.mesh.hexmesh as hexmesh
     import mimpy.mfd.mfd as mfd
     import mimpy.models.singlephase.singlephase as singlephase 

Here we find the three main components of Mimpy: (1) **mesh**, (2) **matrix** and (3) **model**. The mesh 
we'll be using is a structured hexahedral mesh from the :mod:`mimpy.mesh.hexmesh` module.
For matrix construction, we use the :mod:`mimpy.mfd.mfd` module. And finally, the model 
we use is in the :mod:`mimpy.models.singlephase` module. 

The first two parts (**mesh** and **matrix**) are common to all models, and the last part 
is specific to the single-phase model. 

(1) **mesh**


The first thing we'll need to do is construct the mesh. We start by initiating 
a mesh object:

.. code-block:: python

     res_mesh = hexmesh.HexMesh()

This initiates an empty hexahderal mesh. Actual construction is done with the 
*build_mesh* routine. In this case,  
:meth:`mimpy.mesh.hexmesh.HexMesh.build_mesh` takes 7 required parameters. 
The first three are the number of cells in each dimension (x, y, z). 
We want our mesh to be 10 x 10 x 10 cells. The next three entries are the dimensions of the domain
(size of the domain) in (x, y, z). Note that Mimpy assumes SI units for all 
parameters. We'll set our problem to 100m x 100m x 100m.  
The last parameter is **K**,  which is a function mapping location to a permeability 
tensor. We set a  constant permeability of 1.e-12m :math:`^2` by defining **K** as:

.. code-block:: python

    def K(p, i, j, k):
        return 1.e-12*np.eye(3)

**K** takes in *p*, which is an ndarray coordinate, as well as an *(i, j, k)* cell index. It returns 
a 3 x 3 ndarray representing the permeability tensor. 

The build_mesh command looks like:

.. code-block:: python
    
    res_mesh.build_mesh(10, 10, 10, 100., 100., 100., K)

We now have a rectangular 3D mesh we can use! 

(2) **matrix**


We next need to define an MFD objects for constructing the matrices. This is surprisingly 
short:

.. code-block:: python

    res_mfd = mfd.MFD()


(3) **model**


Finally, we can start working on the model. First, we initate a 
single-phase object:

.. code-block:: python
    
    res_singlephase = singlephase.SinglePhase()


We next link our mesh and MFD objects to it:

.. code-block:: python 
 
    res_singlephase.set_mesh_mfd(res_mesh, res_mfd)


We now want to set parameters and boundary conditions for our model. The single-phase 
model allows for two kinds of boundary condition, flux and pressure boundaries. 
For this example we set zero
pressure boundaries. In order to simplify boundary assigments, collections faces on the 
boundary of the mesh are identified by boundary markers. The HexMesh class assigns six boundary 
markers, and are associated with the six sides of the domain:

.. image:: hexboundary.svg
    :align: center 

For this example, we'll set all six to a pressure of zero using the 
:meth:`mimpy.models.singlephase.SinglePhase.apply_pressure_boundary_from_function` routine:

.. code-block:: python 

    res_singlephase.apply_pressure_boundary_from_function(0, lambda p: 0.)
    res_singlephase.apply_pressure_boundary_from_function(1, lambda p: 0.)
    res_singlephase.apply_pressure_boundary_from_function(2, lambda p: 0.)
    res_singlephase.apply_pressure_boundary_from_function(3, lambda p: 0.)
    res_singlephase.apply_pressure_boundary_from_function(4, lambda p: 0.)
    res_singlephase.apply_pressure_boundary_from_function(5, lambda p: 0.)

We next set standard parameters for single-phase flow. These include 
initial pressure, porosity, reference density, reference pressure, 
compressibility and viscosity:

.. code-block:: python 

     res_singlephase.set_initial_pressure(np.zeros(res_mesh.get_number_of_cells()))
     res_singlephase.set_porosities(np.array([.3]*res_mesh.get_number_of_cells()))
     res_singlephase.set_ref_density(1.)
     res_singlephase.set_ref_pressure(0.)
     res_singlephase.set_compressibility(1.e-8)
     res_singlephase.set_viscosity(8.90e-4)


Since we've set the boundaries and initial conditions to zero, running the simulator would 
be very boring. In order to get things moving around, we're going to add two wells to the model. 
Wells are assigned to cells, and there are a few different ways of finding cell numbers. 
For this example, we use the :meth:`mimpy.mesh.Mesh.find_cell_near_point` routine in order to 
give us the cell index of the cell closest to a given coordinate:

.. code-block:: python 

    well_location1 = res_mesh.find_cell_near_point(np.array([50., 75., 50.]))
    well_location2 = res_mesh.find_cell_near_point(np.array([50., 25., 50.]))

Using these cell indices, we can now set a rate specified well:

.. code-block:: python 

    res_singlephase.add_point_rate_well(.1, well_location1, "WELL1")
    res_singlephase.add_point_rate_well(.1, well_location2, "WELL2")


Now all our model parameters have been set. The last part is to assign a time-step 
information:

.. code-block:: python 

    res_singlephase.set_time_step_size(100.)
    res_singlephase.set_number_of_time_steps(20)

as well as output frequency:

.. code-block:: python 

    res_singlephase.set_output_frequency(2)

Finally, we tell it to initalize the system and start running:

.. code-block:: python 

    res_singlephase.initialize_system()
    res_singlephase.start_solving()

















