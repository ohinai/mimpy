
#
#
# Two Phase Buckely-Leverett from:
#
# Numerical modeling of two-phase flow in heterogeneous
# permeable media with different capillary pressures
# Hussein Hoteit Abbas Firoozabadi
#

import mimpy.mesh.hexmesh as hexmesh
import mimpy.mfd.mfd as mfd
import mimpy.models.twophase.twophase as twophase 
import numpy as np

mfd = mfd.MFD()
mesh = hexmesh.HexMesh()

mfd.set_mesh(mesh)

def res_k(point, i, j, k):
    return 1.e-12 * np.eye(3) * 1.e-3

mesh.build_mesh(301, 2, 2, res_k, 80., 1., 1.)

## Using SI Units:
##  viscosity        Kg/(m s)
##  permeability     m^2
##  density          Kg/(m^3)
##  time             s
## 


res_twophase = twophase.TwoPhase()
res_twophase.set_mesh_mfd(mesh, mfd)

res_twophase.apply_flux_boundary_from_function(0, lambda p:np.array([0.,0.,0.]))
res_twophase.apply_pressure_boundary_from_function(1, lambda p:0.)
res_twophase.apply_flux_boundary_from_function(2, lambda p:np.array([0.,0.,0.]))
res_twophase.apply_flux_boundary_from_function(3, lambda p:np.array([0.,0.,0.]))
res_twophase.apply_flux_boundary_from_function(4, lambda p:np.array([0.,0.,0.]))
res_twophase.apply_flux_boundary_from_function(5, lambda p:np.array([0.,0.,0.]))


res_twophase.set_initial_p_o(np.array([100.]*mesh.get_number_of_cells()))
res_twophase.set_initial_u_t(np.array([0.]*mesh.get_number_of_faces()))

res_twophase.set_initial_s_w(np.array([0.]*mesh.get_number_of_cells()))
res_twophase.set_porosities(np.array([.2]*mesh.get_number_of_cells()))

res_twophase.set_viscosity_water(8.90e-4)
res_twophase.set_viscosity_oil(8.90e-4*2.)

res_twophase.set_compressibility_water(0.)
res_twophase.set_compressibility_oil(0.)

res_twophase.set_ref_density_water(1000.)
res_twophase.set_ref_density_oil(1000.)

res_twophase.set_ref_pressure_oil(0.)
res_twophase.set_ref_pressure_water(0.)

res_twophase.set_residual_saturation_water(.0)
res_twophase.set_residual_saturation_oil(.2)

def krw(se):
    return se**2
    return .9 * se*se

def kro(se):
    return (1.-se)**2
    return .5 *(1.- se)**2

res_twophase.set_kro(kro)
res_twophase.set_krw(krw)

res_twophase.set_time_step_size(200000.)

res_twophase.add_rate_well(3./(60.*60.*24), 0., 0, "WELL1")

res_twophase.set_output_frequency(10)
res_twophase.set_number_of_time_steps(1000)

res_twophase.initialize_system()
res_twophase.start_solving()

    
        
    
            
    

                
                

                
                
                

    
