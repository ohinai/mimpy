import sys

import mimpy.mesh.hexmeshwmsfracs as mesh
import mimpy.mfd.mfd as mfd
import numpy as np

# Include relative path for mimpy library.


res_mfd = mfd.MFD()
res_mfd.set_compute_diagonality(True)
res_mfd.set_m_e_construction_method(0)


# Define the permeability function
def K(p, i, j, k):
    return np.eye(3)


# set the mesh and an instance of the HexMesh class
res_mesh = mesh.HexMeshWMSFracs()


# The modification function is applied to the points of the mesh.
# In this case no change is applied.
def mod_function(p, i, j, k):
    return p


frac_file = open("fracs.dat")
frac_file.readline()
frac_list = []

count = 0
for line in frac_file:
    line_split = line.split()
    new_frac = mesh.FracData()

    new_frac.azimuth = float(line_split[0]) / 180.0 * np.pi

    new_frac.dip = float(line_split[1]) / 180.0 * np.pi

    new_frac.a = float(line_split[2]) / 2.0
    new_frac.b = float(line_split[3]) / 2.0

    new_frac.id = count
    count += 1

    frac_list.append(new_frac)
    new_frac.normal = new_frac.get_normal()

    point_x = float(line_split[4])
    point_y = float(line_split[5])
    point_z = float(line_split[6])

    new_frac.center = np.array([point_x, point_y, point_z])

    new_frac.generate_polygon(23)

res_mesh.build_mesh(22, 22, 22, 300.0, 300.0, 300.0, K, mod_function)


count = 1
fracture_faces_list = []
for frac in frac_list:
    frac.output_vtk("frac" + str(count))
    count += 1
    fracture_faces_list.append(res_mesh.add_fractures(frac))

count = 0
for key in res_mesh.fracture_faces_multi:
    fracture_faces = res_mesh.fracture_faces_multi[key]
    count += 1
    res_mesh.output_vtk_faces("faces_" + str(count), list(fracture_faces))
    res_mesh.build_frac_from_faces(list(fracture_faces))

res_mfd.set_mesh(res_mesh)

res_mfd.apply_dirichlet_from_function(0, lambda x: 0.0)
res_mfd.apply_dirichlet_from_function(1, lambda x: 10.0)
res_mfd.apply_dirichlet_from_function(2, lambda x: 0.0)
res_mfd.apply_dirichlet_from_function(3, lambda x: 0.0)
res_mfd.apply_dirichlet_from_function(4, lambda x: 0.0)
res_mfd.apply_dirichlet_from_function(5, lambda x: 0.0)

# Build the LHS and RHS.
res_mfd.build_lhs()
res_mfd.build_rhs()

# Solve the linear system.
res_mfd.solve()

# Output the solution in the vtk format. It will be saved in
# the file "hexmes_example_1.vtk".
res_mesh.output_vtk_mesh(
    "hexmesh_example_1",
    [res_mfd.get_pressure_solution(), res_mesh.get_cell_domain_all()],
    ["MFDPressure", "DOMAIN"],
)
