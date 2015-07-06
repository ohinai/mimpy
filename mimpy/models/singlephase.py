from __future__ import absolute_import
from __future__ import print_function


import numpy as np
from scipy import sparse, diag, integrate, special 
import math as math
import scipy.sparse.linalg.dsolve as dsolve
import os
from multiprocessing import Pool
import itertools

import mimpy.mfd.mfd as mfd
from six.moves import range

class SinglePhase():
    """ The class relies on the Mesh and MFD libraries
    to solve time dependent slighly-compressible
    flow problems (heat equation).
    .. math::
        \phi \frac{\partial \rho}{\partial t}  =
        -\nabla \cdot \lef( \frac{\rho}{\mu} K (\nabla p)\right) + q

    """
    def __init__(self):
        self.mesh = None
        self.mfd = None

        self.initial_pressure = None
        self.current_pressure = None

        # Reservoir properties:
        self.porosities = None

        # Fluid properties:
        self.viscosity = None

        self.ref_pressure = None
        self.ref_density = None
        self.compressibility = None

        # Matrix M in coo format 
        # without density or viscosity 
        # data. 
        self.m_x_coo_data = None
        self.m_e_locations = None
        self.m_x_coo_length = None

        self.c_start = None

        # Problem LHS in coo format. 
        self.lhs_coo = None

        # RHS from MFD class. 
        self.rhs_mfd = None

        # Current RHS. 
        self.rhs_current = None

        # Time step size. 
        self.delta_t = 1.
        self.number_of_time_steps = 1

        # Sets how often to output 
        # solution by time step number. 
        self.output_frequency = 1

        # Rate wells cell location
        self.rate_wells = []
        self.rate_wells_rate = []
        self.rate_wells_name = []

        # Model Name 
        self.model_name = "model"
        
    def set_model_name(self, name):
        """ Sets model name used for output. 
        """
        self.model_name = name

    def set_output_frequency(self, frequency):
        """ Sets how often to output the solution 
        by iteration number. x
        """
        self.output_frequency = frequency

    def set_mesh(self, mesh):
        """ Sets the computational mesh 
        to be used. 
        """
        self.mesh = mesh
        self.mfd = mfd.MFD()
        self.mfd.set_mesh(mesh)
        
    def set_compressibility(self, compressibility):
        """ Sets fluid compressilibity. 
        """
        self.compressibility = compressibility
    
    def set_ref_density(self, ref_density):
        """ Sets reference density of fluid at p0. 
        """
        self.ref_density = ref_density

    def set_ref_pressure(self, ref_pressure):
        """ Set reference pressure.
        """
        self.ref_pressure = ref_pressure

    def set_viscosity(self, viscosity):
        """ Set fluid viscosity. 
        """
        self.viscosity = viscosity

    def set_initial_pressure(self, pressures):
        """ Sets the pressure distributionat 
        time 0. 
        """
        self.initial_pressure = pressures
        self.current_pressure = pressures
        # Assuming zero initial velocity for now. 
        self.current_velocity = np.zeros(self.mesh.get_number_of_faces())
        
    def set_porosities(self, porosities):
        """ Sets cell porosities.
        """ 
        self.porosities = porosities
    
    def set_porosity_by_cell(self, cell_index, porosity):
        """ Sets porosity for cell_index. 
        """
        self.porosities[cell_index] = porosity

    def add_rate_well(self, injection_rate, cell_index, well_name):
        """ Adds rate specified well at the center of cell_index.
        Returns the index of the new rate well. 
        The units of rate are kg/s. 
        """
        raise NameError("add_rate_well not yet implemented")
    
    def add_point_rate_well(self, injection_rate, cell_index, well_name):
        """ Adds rate specified point source at the center of cell_index.
        Returns the index of the new rate well. 
        The units of rate are kg/s. 
        """
        self.rate_wells.append(cell_index)
        self.rate_wells_rate.append(injection_rate)
        self.rate_wells_name.append(well_name)
        return len(self.rate_wells)-1

    def set_time_step_size(self, delta_t):
        """ Sets time step size. 
        """
        self.delta_t = delta_t

    def set_number_of_time_steps(self, number_of_time_steps):
        """ Set number of time steps taken. 
        """
        self.number_of_time_steps = number_of_time_steps

    def apply_pressure_boundary_from_function(self, 
                                              boundary_marker, 
                                              p_function):
        """ Applies static pressure (dirichlet) boundary 
        conditions from function. 
        """
        self.mfd.apply_dirichlet_from_function(boundary_marker, 
                                               p_function)

    def apply_flux_boundary_from_function(self,
                                              boundary_marker, 
                                              f_function):
        """ Applies static flux (neumann) boundary 
        conditions from function. 
        """
        self.mfd.apply_neumann_from_function(boundary_marker,
                                             f_function)
        
    def initialize_system(self):
        """ Constructs the initial matrices 
        used to construct the saddle-point 
        problem. 
        """
        self.mfd.set_mesh(self.mesh)
        [[div_data, div_row, div_col], 
         [div_t_data, div_t_row, div_t_col]] = self.mfd.build_div()
        [self.m_x_coo_data, 
         m_x_coo_row, 
         m_x_coo_col] = self.mfd.build_m(save_update_info=True)

        self.m_x_coo_length = len(self.m_x_coo_data)
        
        # The data for the bottom right should be zeros. 
        [c_data, c_row, c_col] = self.mfd.build_bottom_right()
        
        [coupling_data, coupling_row, coupling_col] = self.mfd.build_coupling_terms()

        lhs_data = self.m_x_coo_data
        lhs_row = m_x_coo_row
        lhs_col = m_x_coo_col
        
        lhs_data += div_data
        lhs_row += div_row
        lhs_col += div_col

        lhs_data += div_t_data
        lhs_row += div_t_row
        lhs_col += div_t_col        
        
        self.c_start = len(lhs_data)
        
        lhs_data += c_data
        lhs_row += c_row
        lhs_col += c_col 

        self.c_end = len(c_data)

        lhs_data += coupling_data
        lhs_row += coupling_row
        lhs_col += coupling_col

        # Convert m_x_coo_data to numpy array. 
        self.m_x_coo_data = np.array(self.m_x_coo_data)

        self.lhs_coo = sparse.coo_matrix((np.array(lhs_data), 
                                          (np.array(lhs_row), 
                                           np.array(lhs_col))))

        # RHS construction is for Neumann and Dirichlet 
        # boundaries specified by the mesh. 
        self.rhs_mfd = self.mfd.build_rhs()

    def time_step_output(self, current_time, time_step):
        """ Function to be defined by user 
        that will be run at each output interval
        during time-stepping. The function 
        is intended for complex output. 
        """
        pass
    
    def start_solving(self):
        """ Starts solving the problem. 
        """
        self.mesh.output_vtk_mesh(self.model_name + "0", 
                                  [self.current_pressure, 
                                   self.mesh.get_cell_domain_all()], 
                                  ["pressure", "domain"])

        self.time_step_output(0., 0)

        m_multipliers = np.ones(self.mesh.get_number_of_cells())

        for time_step in range(1,self.number_of_time_steps+1):
            rhs_current = np.zeros(self.mfd.get_number_of_dof())        
            rhs_current += self.rhs_mfd
            
            current_time = time_step*self.delta_t
            print(time_step)
            
            for cell_index in range(self.mesh.get_number_of_cells()):
                density = -self.ref_pressure
                density += self.current_pressure[cell_index]
                density *= self.compressibility
                density += 1.
                density *= self.ref_density

                # We multiply by the inverse of \frac{\rho}{\mu}
                m_multipliers[cell_index] = self.viscosity/density

                c_entry = self.compressibility
                c_entry *= self.porosities[cell_index]
                c_entry /= self.delta_t
                c_entry *= self.mesh.get_cell_volume(cell_index)

                rhs_current[self.mesh.get_number_of_faces()+
                            cell_index] += c_entry*self.current_pressure[cell_index]
                
                self.lhs_coo.data[self.c_start+cell_index] = c_entry

            for [index, cell_index] in enumerate(self.rate_wells):
                rhs_current[self.mesh.get_number_of_faces()+cell_index] += \
                    self.rate_wells_rate[index]
                    
            self.mfd.update_m(self.lhs_coo.data[:self.m_x_coo_length], m_multipliers)
            
            solution = dsolve.spsolve(self.lhs_coo.tocsr(), rhs_current)
            self.current_pressure = solution[self.mesh.get_number_of_faces():]
            self.current_velocity = solution[:self.mesh.get_number_of_faces()]

            if time_step%self.output_frequency == 0:
                self.mesh.output_vtk_mesh(self.model_name+str(time_step), 
                                          [self.current_pressure, 
                                           self.mesh.get_cell_domain_all()],
                                          ["pressure", "domain"])

                self.time_step_output(current_time, time_step)

             
            
                

                
            
                
                
                


            
            
    
                

    
    
