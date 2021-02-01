
import mimpy.mesh.mesh as mesh
import mimpy.mfd.mfd as mfd
import numpy as np

import pickle

res_mfd = mfd.MFD()
res_mfd.set_compute_diagonality(True)
res_mfd.set_m_e_construction_method(7)

def u(p):
    return p[0]

#set the mesh as an instance of the HexMesh class
res_mesh = pickle.load(open("fault"))

res_mesh.use_face_shifted_centroid()
res_mesh.set_face_shifted_to_tpfa_all()

#Connect the MFD instance to the new mesh. 
res_mfd.set_mesh(res_mesh)

#Apply Dirichlet boundary conditions to all 6 faces. 
res_mfd.apply_neumann_from_function(0, lambda p:np.array([0.,0.,0.]))
res_mfd.apply_neumann_from_function(1, lambda p:np.array([0.,0.,0.]))
res_mfd.apply_dirichlet_from_function(2, lambda p:u(p))

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


