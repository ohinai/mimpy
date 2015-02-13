

import numpy as np
from scipy import sparse, diag, integrate, special 
import math as math
import scipy.sparse.linalg.dsolve as dsolve
import os
from multiprocessing import Pool
import itertools

class SinglePhase():
    """
    The class relies on the Mesh and MFD libraries 
    to solve time dependent slighly-compressible 
    flow problems. 
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

        # Parameters for sovling the problem analytically.
        self.compute_analytical_solution = False

        # Rate wells cell location
        self.rate_wells = []
        self.rate_wells_rate = []
        self.rate_wells_name = []

        # Model Name 
        self.model_name = "model"

        # Max analytical iterations:
        self.m_max = 100
        self.n_max = 100
        self.w_max = 100
        
        # Number of concurrent threads to run 
        # for computing analytical solution. 
        self.thread_pool = 4
        
    def set_model_name(self, name):
        """
        Sets model name used for output. 
        """
        self.model_name = name

    def set_output_frequency(self, frequency):
        """ Sets how often to output the solution 
        by iteration number. x
        """
        self.output_frequency = frequency

    def set_mesh(self, mesh):
        """
        Sets the computational mesh 
        to be used. 
        """
        self.mesh = mesh
        
    def set_mfd(self, mfd):
        """
        Sets the instance of 
        MFD to be used. 
        """
        self.mfd = mfd
    
    def set_compressibility(self, compressibility):
        """
        Sets fluid compressilibity. 
        """
        self.compressibility = compressibility
    
    def set_ref_density(self, ref_density):
        """
        Sets reference density of fluid at p0. 
        """
        self.ref_density = ref_density

    def set_ref_pressure(self, ref_pressure):
        """
        Set reference pressure.
        """
        self.ref_pressure = ref_pressure

    def set_viscosity(self, viscosity):
        """
        Set fluid viscosity. 
        """
        self.viscosity = viscosity

    def set_initial_pressure(self, pressures):
        """
        Sets the pressure distributionat 
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
        """
        Adds rate specified well at the center of cell_index.
        Returns the index of the new rate well. 
        The units of rate are kg/s. 
        """
        raise NameError("add_rate_well not yet implemented")
    
    def add_point_rate_well(self, injection_rate, cell_index, well_name):
        """
        Adds rate specified point source at the center of cell_index.
        Returns the index of the new rate well. 
        The units of rate are kg/s. 
        """
        self.rate_wells.append(cell_index)
        self.rate_wells_rate.append(injection_rate)
        self.rate_wells_name.append(well_name)
        return len(self.rate_wells)-1

    def set_time_step_size(self, delta_t):
        """
        Sets time step size. 
        """
        self.delta_t = delta_t

    def set_number_of_time_steps(self, number_of_time_steps):
        """
        Set number of time steps taken. 
        """
        self.number_of_time_steps = number_of_time_steps
        
    def initialize_system(self):
        """
        Constructs the initial matrices 
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
        if self.compute_analytical_solution:
            self.mesh.output_vtk_mesh(self.model_name + "0", 
                                      [self.current_pressure, 
                                       self.initial_pressure], 
                                      ["pressure", "analytical"])
        else:
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
            print time_step
            
            for cell_index in range(self.mesh.get_number_of_cells()):
                density = -self.ref_pressure
                density += self.current_pressure[cell_index]
                density *= self.compressibility
                density += 1.
                density *= self.ref_density

                # We multiply by the inverse of \frac{\rho}{\mu}
                m_multipliers[cell_index] = self.viscosity

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
                if self.compute_analytical_solution:
                    self.mesh.output_vtk_mesh(self.model_name+str(time_step), 
                                              [self.current_pressure, 
                                               self.analytical_solution_at_centroids(current_time)],
                                              ["pressure", "analytical"])
                else:
                    self.mesh.output_vtk_mesh(self.model_name+str(time_step), 
                                              [self.current_pressure, 
                                               self.mesh.get_cell_domain_all()],
                                              ["pressure", "domain"])

                self.time_step_output(current_time, time_step)

    def set_compute_analytical_solution(self, setting):
        """
        Sets whether the analytical solution is computed. 
        """
        self.compute_analytical_solution = setting
        
    def theta_3(self, x_in, t_in):
        """
        Elliptic theta function of the third kind. 
        \Theta_3(\pi x, e^{-\pi^2t}) 
        """
        total = 0.
        pi_sqrd = np.pi**2
        if np.e**(-(pi_sqrd)*t_in)>(1./np.pi):
            for n in range(1, self.theta_loop_max):
                total += np.e**(-n*n*pi_sqrd*t_in)*np.cos(2.*n*np.pi*x_in)
            total *= 2.
            total += 1.
            return total
        else:
            for n in range(-self.theta_loop_max+1, self.theta_loop_max):
                total += np.e**(-((x_in+n)**2.)/t_in)
            total *= 1./np.sqrt(np.pi*t_in)
            return total 

    def theta_3_prime(self, x_in, t_in):
        """
        Derivative of the elliptic theta function of 
        the third kind. 
        \Theta^{'}_3(\pi x, e^{-\pi^2t}) 
        """
        total = 0.
        pi_sqrd = np.pi**2
        if np.e**(-(pi_sqrd)*t)>(1./np.pi):
            for n in range(1, self.theta_loop_max):
                total += n*np.e**(-n*n*pi_sqrd*t_in)*np.sin(2.*n*np.pi*x_in)
            total *= -4.*np.pi
            return total
        else:
            for n in range(-self.theta_loop_max+1, self.theta_loop_max):
                total += (x+n)*np.e**(-((x_in+n)**2.)/t_in)
            total *= -2./np.sqrt(np.pi*t_in**3)
            return total
        
    def analytical_solution_at_centroids(self, t_in):
        """
        Returns a list of exact pressures 
        at the centroids of the cells. 
        """
        n_x = self.mesh.get_cell_k(0)[0,0]
        n_x /= self.porosities[0]
        n_x /= self.viscosity
        n_x /= self.compressibility

        n_y = self.mesh.get_cell_k(0)[1,1]
        n_y /= self.porosities[0]
        n_y /= self.viscosity
        n_y /= self.compressibility

        n_z = self.mesh.get_cell_k(0)[2,2]
        n_z /= self.porosities[0]
        n_z /= self.viscosity
        n_z /= self.compressibility
        
        a = self.mesh.get_dim_x()
        b = self.mesh.get_dim_y()
        d = self.mesh.get_dim_z()
        
        print "t_in", t_in
        pool = Pool(processes=self.thread_pool)
        
        pressure_vector = np.zeros(self.mesh.get_number_of_cells())

        # Find out the boundary conditions 
        # for appropriate analytical solution. 
        # Currently can handle either all 
        # Dirichlet or all Neumann.

        if self.mesh.get_number_of_dirichlet_faces() > 0:
            all_dirichlet = True 
            
        else:
            all_dirichlet = False
        
        for (well, well_cell_index) in enumerate(self.rate_wells):
            well_location = self.mesh.get_cell_real_centroid(well_cell_index)
            
            x_well = well_location[0]
            y_well = well_location[1]
            z_well = well_location[2]
            
            factor = self.rate_wells_rate[well]
            factor /= 8.
            factor /= self.compressibility
            factor /= self.porosities[0]
            factor /= a*b*d

            if all_dirichlet:
                pressure_vector += factor*np.array(
                    pool.map(analytical_solution_d, 
                             zip(self.mesh.get_all_cell_real_centroids(),
                                 itertools.repeat(t_in, self.mesh.get_number_of_cells()),
                                 itertools.repeat(x_well, self.mesh.get_number_of_cells()),
                                 itertools.repeat(y_well, self.mesh.get_number_of_cells()),
                                 itertools.repeat(z_well, self.mesh.get_number_of_cells()),
                                 itertools.repeat(n_x, self.mesh.get_number_of_cells()),
                                 itertools.repeat(n_y, self.mesh.get_number_of_cells()),
                                 itertools.repeat(n_z, self.mesh.get_number_of_cells()),
                                 itertools.repeat(a, self.mesh.get_number_of_cells()),
                                 itertools.repeat(b, self.mesh.get_number_of_cells()),
                                 itertools.repeat(d, self.mesh.get_number_of_cells()),
                                 itertools.repeat(self.n_max, self.mesh.get_number_of_cells()),
                                 itertools.repeat(self.m_max, self.mesh.get_number_of_cells()),
                                 itertools.repeat(self.w_max, self.mesh.get_number_of_cells()))))
            else:
                pressure_vector += factor*np.array(
                    pool.map(analytical_solution_n, 
                             zip(self.mesh.get_all_cell_real_centroids(),
                                 itertools.repeat(t_in, self.mesh.get_number_of_cells()),
                                 itertools.repeat(x_well, self.mesh.get_number_of_cells()),
                                 itertools.repeat(y_well, self.mesh.get_number_of_cells()),
                                 itertools.repeat(z_well, self.mesh.get_number_of_cells()),
                                 itertools.repeat(n_x, self.mesh.get_number_of_cells()),
                                 itertools.repeat(n_y, self.mesh.get_number_of_cells()),
                                 itertools.repeat(n_z, self.mesh.get_number_of_cells()),
                                 itertools.repeat(a, self.mesh.get_number_of_cells()),
                                 itertools.repeat(b, self.mesh.get_number_of_cells()),
                                 itertools.repeat(d, self.mesh.get_number_of_cells()),
                                 itertools.repeat(self.n_max, self.mesh.get_number_of_cells()),
                                 itertools.repeat(self.m_max, self.mesh.get_number_of_cells()),
                                 itertools.repeat(self.w_max, self.mesh.get_number_of_cells()))))
                
            
        return pressure_vector

    
    def analytical_solution(self, x_in, y_in, z_in, t_in):
        """
        Analytical solution to time dependent 
        slightly compressible problem over a 3D domain.
        The soluion is based on the analytical solutions 
        found in "The Diffusion Handbook" By 
        R. K. M. Thambynayagam.

        Distance: meters
        Time: seconds
        Permeability: m^2
        viscosity: Pa*s
        Compressibility: 1/Pa
        
        Using relation G.18 on Page 1898, 
        we have:
        \sum&{\infty}_{n=1} sin(nx)sin(ny)e^{-\beta n^2} = 
        1/4[\Theta_3{1/2(x-y), e^{-\Beta}} - \Theta_3{1/2(x+y), e^{-\Beta}}] (\Beta > 0) =
        G(x, y, -\Beta)
        
        For the intergral 
        G(\pi x/a, \phi x_0/a, -(\pi/a)^2 \eta_x \tau)
        G(\pi y/b, \phi y_0/b, -(\pi/b)^2 \eta_y \tau)
        G(\pi z/d, \phi z_0/d, -(\pi/d)^2 \eta_z \tau) = 
        64*\sum_n \sum_m \sum_w 
        sin(n\pi x/a)sin(n\phi x_0/a)
        sin(m\pi y/b)sin(m\phi y_0/b)
        sin(w\pi z/d)sin(w\phi z_0/d)
        e^{-\tau((\pi n/a)^2 \eta_x+(\pi m/b)^2 \eta_y+(\pi w/c)^2 \eta_z)}
        
        Integrating in \tau, 

        \int^{t}_0 64\sum_n \sum_m \sum_w 
        sin(n\pi x/a)sin(n\phi x_0/a)
        sin(m\pi y/b)sin(m\phi y_0/b)
        sin(w\pi z/d)sin(w\phi z_0/d)
        e^{-\tau((\pi n/a)^2 \eta_x+(\pi m/b)^2 \eta_y+(\pi w/c)^2 \eta_z)}d\tau=
        64\sum_n \sum_m \sum_w 
        sin(n\pi x/a)sin(n\phi x_0/a)
        sin(m\pi y/b)sin(m\phi y_0/b)
        sin(w\pi z/d)sin(w\phi z_0/d)
        \int^{t}_0 e^{-\tau((\pi n/a)^2 \eta_x+(\pi m/b)^2 \eta_y+(\pi w/c)^2 \eta_z)}d\tau= 
        64 \sum_n \sum_m \sum_w 
        sin(n\pi x/a)sin(n\phi x_0/a)
        sin(m\pi y/b)sin(m\phi y_0/b)
        sin(w\pi z/d)sin(w\phi z_0/d)
        \frac{-1}{(\pi n/a)^2\eta_x+(\pi m/b)^2\eta_y+(\pi w/c)^2\eta_z}
        \left(e^{-t((\pi n/a)^2\eta_x+(\pi m/b)^2\eta_y+(\pi w/c)^2\eta_z)}-1\right)= 
        """
        injection_rate = 1.
        
        compressibility = 1.e-8
        viscosity =  8.90e-4
        permeability = 10.e-12
        porosity = .2

        n_x = permeability 
        n_x /= porosity
        n_x /= viscosity
        n_x /= compressibility

        n_y = n_x
        n_z = n_x
        
        x_well = 5.
        y_well = 5.
        z_well = 5.

        n_x = 1.
        n_y = 1.
        n_z = 1.
        
        a = 10.
        b = 10.
        d = 10.

        # 64 \sum_n \sum_m \sum_w 
        # sin(n\pi x/a)sin(n\phi x_0/a)
        # sin(m\pi y/b)sin(m\phi y_0/b)
        # sin(w\pi z/d)sin(w\phi z_0/d)
        # \frac{-1}{(\pi n/a)^2\eta_x+(\pi m/b)^2\eta_y+(\pi w/c)^2\eta_z}
        # \left(e^{-t((\pi n/a)^2\eta_x+(\pi m/b)^2\eta_y+(\pi w/c)^2\eta_z)}-1\right) 
        
        pressure = 0.

        for n in range(1, 100):
            current_n = np.sin(n*np.pi*x_in/a)
            current_n *= np.sin(n*np.pi*x_well/a)
            for m in range(1, 100):
                current_n_m = current_n*np.sin(m*np.pi*y_in/b)
                current_n_m *= np.sin(m*np.pi*y_well/b)
                for w in range(1, 100):
                    current_value = current_n_m*np.sin(w*np.pi*z_in/d)
                    current_value *= np.sin(w*np.pi*z_well/d)
                    current_value /= (np.pi*n/a)**2*n_x+(np.pi*m/b)**2*n_y+(np.pi*w/d)**2*n_z
                    current_value *= np.exp(-t_in*((np.pi*n/a)**2*n_x+
                                                   (np.pi*m/b)**2*n_y+
                                                   (np.pi*w/d)**2*n_z))-1.
                    pressure += -current_value
        pressure *= .001/(8.*porosity*compressibility*a*b*d)
            
        return pressure

        def func_1(t):
            """
            Function integrated over time used in calculating 
            the analytical solution.
            """
            return_value = 1.
            return_value *= injection_rate

            return_value *= (self.theta_3((x_in-well_x)/(2.*a), n_x*t/(a*a))-
                             self.theta_3((x_in+well_x)/(2.*a), n_x*t/(a*a)))

            return_value *= (self.theta_3((y_in-well_y)/(2.*b), n_y*t/(b*b))-
                             self.theta_3((y_in+well_y)/(2.*b), n_y*t/(b*b)))

            return_value *= (self.theta_3((z_in-well_z)/(2.*d), n_z*t/(d*d))-
                             self.theta_3((z_in+well_z)/(2.*d), n_z*t/(d*d)))
            
            return return_value

        #for current_t in np.arange(.00001, t_in, 10.):
        #    pressure += func_1(current_t)/1.
        pressure += func_1(current_t)/1.
            
#        pressure += integrate.quad(func_1, 0, t_in)[0]/(8.*porosity*compressibility*a*b*d)
        
        return pressure
        

             
            
                

                
            
                
                
                


            
            
    
                

    
    
