
import mimpy.mesh.hexmesh as hexmesh
import mimpy.mfd.mfd as mfd
import numpy as np

res_mfd = mfd.MFD()
res_mfd.set_compute_diagonality(True)
res_mfd.set_m_e_construction_method(0)

#Define the permeability function
def K(p, i, j, k):
    return np.eye(3)

#The exact solution.
def u(p):
    return p[0]**2 + p[1]**2 + p[2]**2

#The forcing function for the exact solution. 
def f(x):
    return -6.

#set the mesh and an instance of the HexMesh class
res_mesh = hexmesh.HexMesh()

#The modification function is applied to the points of the mesh. 
#In this case no change is applied. 
def mod_function(p, i, j, k):
    return p

res_mesh.build_mesh(10, 10,  10, K, 1., 1., 1., mod_function)

#Connect the MFD instance to the new mesh. 
res_mfd.set_mesh(res_mesh)

#Apply Dirichlet boundary conditions to all 6 faces. 
res_mfd.apply_dirichlet_from_function(0, lambda p:u(p))
res_mfd.apply_dirichlet_from_function(1, lambda p:u(p))
res_mfd.apply_dirichlet_from_function(2, lambda p:u(p))
res_mfd.apply_dirichlet_from_function(3, lambda p:u(p))
res_mfd.apply_dirichlet_from_function(4, lambda p:u(p))
res_mfd.apply_dirichlet_from_function(5, lambda p:u(p))

#Apply the forcing function f. 
res_mfd.apply_forcing_from_function(f)

#Build the LHS and RHS. 
res_mfd.build_lhs()
res_mfd.build_rhs()

#Solve the linear system. 
res_mfd.solve()

#Output the solution in the vtk format. It will be saved in 
#the file "hexmes_example_1.vtk". 
res_mesh.output_vtk_mesh("hexmesh_example_1", 
                         [res_mfd.get_pressure_solution(), 
                          res_mfd.get_analytical_pressure_solution(u), 
                          res_mfd.get_diagonality()],
                         ["MFDPressure", "AnalyticalPressure", "ORTHO"])


