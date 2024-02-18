""" Two Phase Buckely-Leverett problem.
"""

import mimpy.mesh.hexmesh as hexmesh
import mimpy.models.twophase as twophase
import numpy as np

mesh = hexmesh.HexMesh()


def res_k(point, i, j, k):
    return 1.0e-12 * np.eye(3) * 1.0e-3


mesh.build_mesh(300, 1, 1, 80.0, 1.0, 1.0, res_k)

res_twophase = twophase.TwoPhase()
res_twophase.set_mesh(mesh)

res_twophase.apply_flux_boundary_from_function(0, lambda p: np.array([0.0, 0.0, 0.0]))
res_twophase.apply_pressure_boundary_from_function(1, lambda p: 0.0)
res_twophase.apply_flux_boundary_from_function(2, lambda p: np.array([0.0, 0.0, 0.0]))
res_twophase.apply_flux_boundary_from_function(3, lambda p: np.array([0.0, 0.0, 0.0]))
res_twophase.apply_flux_boundary_from_function(4, lambda p: np.array([0.0, 0.0, 0.0]))
res_twophase.apply_flux_boundary_from_function(5, lambda p: np.array([0.0, 0.0, 0.0]))

res_twophase.set_initial_p_o(np.array([100.0] * mesh.get_number_of_cells()))

res_twophase.set_initial_s_w(np.array([0.0] * mesh.get_number_of_cells()))
res_twophase.set_porosities(np.array([0.2] * mesh.get_number_of_cells()))

res_twophase.set_viscosity_water(8.90e-4)
res_twophase.set_viscosity_oil(8.90e-4 * 2.0)

res_twophase.set_compressibility_water(0.0)
res_twophase.set_compressibility_oil(0.0)

res_twophase.set_ref_density_water(1000.0)
res_twophase.set_ref_density_oil(1000.0)

res_twophase.set_ref_pressure_oil(0.0)
res_twophase.set_ref_pressure_water(0.0)

res_twophase.set_residual_saturation_water(0.0)
res_twophase.set_residual_saturation_oil(0.2)

res_twophase.set_corey_relperm(2.0, 2.0)

res_twophase.set_time_step_size(200000.0)

res_twophase.add_rate_well(3.0 / (60.0 * 60.0 * 24), 0.0, 0, "WELL1")

res_twophase.set_output_frequency(10)
res_twophase.set_number_of_time_steps(1000)

res_twophase.initialize_system()
res_twophase.start_solving()
