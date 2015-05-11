
import mimpy.mesh.hexmesh as hexmesh
import mimpy.mfd.mfd as mfd
import mimpy.models.singlephase as singlephase 
import numpy as np

#Define the permeability function
def K(p, i, j, k):
    return 1.e-12*np.eye(3)

#The modification function is applied to the points of the mesh. 
#In this case no change is applied. 
def mod_function(p, i, j, k):
    return p

res_mesh = hexmesh.HexMesh()

## Build the mesh 
res_mesh.build_mesh(10, 10, 10, 100., 100., 100., K, mod_function)

## Initialize the MFD class
res_mfd = mfd.MFD()

## Initialize the singlephase model
res_singlephase = singlephase.SinglePhase()

## Assign the mesh and mfd objects to singlephase
res_singlephase.set_mesh_mfd(res_mesh, res_mfd)

# Apply Dirichlet boundary conditions to all 6 faces. 
res_singlephase.apply_pressure_boundary_from_function(0, lambda p: 0.)
res_singlephase.apply_pressure_boundary_from_function(1, lambda p: 0.)
res_singlephase.apply_pressure_boundary_from_function(2, lambda p: 0.)
res_singlephase.apply_pressure_boundary_from_function(3, lambda p: 0.)
res_singlephase.apply_pressure_boundary_from_function(4, lambda p: 0.)
res_singlephase.apply_pressure_boundary_from_function(5, lambda p: 0.)

## Initial pressure 
res_singlephase.set_initial_pressure(np.zeros(res_mesh.get_number_of_cells()))

## Porosity of the rock 
res_singlephase.set_porosities(np.array([.3]*res_mesh.get_number_of_cells()))

## Fluid density 
res_singlephase.set_ref_density(1.)

## Reference pressure for compressibility
res_singlephase.set_ref_pressure(0.)

## Fluid compressibility
res_singlephase.set_compressibility(1.e-8)

## Fluid viscosity 
res_singlephase.set_viscosity(8.90e-4)

## Locate cell index based on location of centroid 
well_location1 = res_mesh.find_cell_near_point(np.array([50., 75., 50.]))
well_location2 = res_mesh.find_cell_near_point(np.array([50., 25., 50.]))

## Assign point rate specified source term. 
res_singlephase.add_point_rate_well(.1, well_location1, "WELL1")
res_singlephase.add_point_rate_well(.1, well_location2, "WELL2")

## Set time-step size
res_singlephase.set_time_step_size(100.)

## Number of time steps to take
res_singlephase.set_number_of_time_steps(20)

## Output frequency for vtk files 
res_singlephase.set_output_frequency(2)

## Initialize the matrices
res_singlephase.initialize_system()

## Start solving
res_singlephase.start_solving()
