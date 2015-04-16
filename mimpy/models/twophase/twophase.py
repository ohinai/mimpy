
import numpy as np
from scipy import sparse, diag
from scipy import interpolate
import scipy.sparse.linalg.dsolve as dsolve

class TwoPhase:
    """ Model class for solving two-phase, immiscible, 
    slightly compressible flow problem based on the 
    Mimetic Finite Difference method. 
    """
    def __init__(self):
        self.mesh = None
        self.mfd = None
        
        self.initial_p_o = None
        self.previous_p_o = None
        self.current_p_o = None

        self.initial_u_t = None
        self.previous_u_t = None
        self.current_u_t = None
        
        self.initial_s_w = None
        self.current_s_w = None

        # Rock properties 
        self.porosities = None
        
        # Fluid properties
        self.viscosity_water = None
        self.viscosity_oil = None
        
        self.ref_pressure_water = None
        self.ref_density_water = None
        self.compressilibity_water = None

        self.ref_pressure_oil = None
        self.ref_density_oil = None
        self.compressilibity_oil = None

        # Relative permeability for water
        self.krw = None

        # Relative permeability for oil
        self.kro = None
        
        self.residual_saturation_water = None
        self.residual_saturation_oil = None

        # Matrix M in coo format 
        # without density or viscosity data.
        self.m_x_coo_data = None
        self.m_e_locations = None
        self.m_x_coo_length = None

        self.c_start = None
        self.c_end = None
        
        # Problem LHS in coo format. 
        self.lhs_coo = None
        
        self.div = None

        # RHS from MFD class. 
        self.rhs_mfd = None
        
        # Saturation boundary conditions
        # saturation_boundaries a dict 
        # that indicates the saturation 
        # value behind Dirichlet boundaires. 
        # boundary_marker => saturation_function(x, y, z)
        self.saturation_boundaries = {}
        
        # Time step size. 
        self.delta_t = 1.
        self.number_of_time_steps = 1
        self.saturation_time_steps = 1
        self.current_time = 0.

        # Rate wells cell location
        self.rate_wells = []
        self.rate_wells_rate_water = []
        self.rate_wells_rate_oil = []
        self.rate_wells_name = []
        self.pressure_files = []

        # Pressure wells cell location
        self.pressure_wells = []
        self.pressure_wells_bhp = []
        self.pressure_wells_pi = []
        self.pressure_wells_name = []
        self.production_files = []
        
        # Files for outputting pressure well production/injection.  
        #self.pressure_well_outputs = []

        # Newton solve parameters. 
        self.newton_threshold = 1.e-6

        # Sets how often to output 
        # solution by time step number. 
        self.output_frequency = 1
        self.prod_output_frequency = 1

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

    def set_mesh_mfd(self, mesh, mfd):
        """ Sets the computational mesh 
        to be used. 
        """
        self.mesh = mesh
        self.mfd = mfd
        self.mfd.set_mesh(mesh)
        
    def set_compressibility_water(self, compressibility_water):
        """ Sets water compressilibity. 
        """
        self.compressibility_water = compressibility_water

    def set_compressibility_oil(self, compressibility_oil):
        """ Sets oil compressilibity. 
        """
        self.compressibility_oil = compressibility_oil
    
    def set_ref_density_oil(self, ref_density_oil):
        """ Sets reference density of oil at reference pressure.
        """
        self.ref_density_oil = ref_density_oil

    def set_ref_density_water(self, ref_density_water):
        """ Sets reference density of water at reference presssure. 
        """
        self.ref_density_water = ref_density_water

    def set_ref_pressure_oil(self, ref_pressure_oil):
        """ Set reference pressure for oil.
        """
        self.ref_pressure_oil = ref_pressure_oil

    def set_ref_pressure_water(self, ref_pressure_water):
        """ Set reference pressure for water.
        """
        self.ref_pressure_water = ref_pressure_water

    def set_viscosity_water(self, viscosity_water):
        """ Set water viscosity. 
        """
        self.viscosity_water = viscosity_water

    def set_kro(self, kro):
        """ Set oil relative permeability function. 
        """
        self.kro = kro

    def set_krw(self, krw):
        """ Set oil relative permeability function. 
        """
        self.krw = krw

    def set_residual_saturation_water(self, residual_saturation_water):
        """ Sets water residual saturation.
        """
        self.residual_saturation_water = residual_saturation_water


    def set_residual_saturation_oil(self, residual_saturation_oil):
        """ Sets oil residual saturation.
        """
        self.residual_saturation_oil = residual_saturation_oil

    def set_viscosity_oil(self, viscosity_oil):
        """ Set oil viscosity. 
        """
        self.viscosity_oil = viscosity_oil

    def sefromsw(self, water_saturation):
        return ((water_saturation-self.residual_saturation_water)/
                (1.-self.residual_saturation_water-self.residual_saturation_oil))

    def water_mobility(self, water_saturation, water_pressure):
        """ Returns water mobility
        """
        mobility = self.krw(self.sefromsw(water_saturation))/self.viscosity_water
        mobility *= self.ref_density_water*(1.+self.compressibility_water*water_pressure)

        return mobility
        
    def oil_mobility(self, water_saturation, oil_pressure):
        """ Returns oil mobility. 
        """   
        mobility = self.kro(self.sefromsw(water_saturation))/self.viscosity_oil
        mobility *= self.ref_density_oil*(1.+self.compressibility_oil*oil_pressure)


        return mobility

    def set_initial_p_o(self, p_o):
        """ Sets the pressure distributionat 
        time 0. 
        """
        self.initial_p_o = p_o
        self.current_p_o = p_o

    def set_initial_u_t(self, u_t):
        """ Sets initial total velocity 
        for system.
        """
        self.current_u_t = u_t
        self.initial_u_t = u_t
        
    def set_initial_s_w(self, s_w):
        """ Sets initial water saturation.
        """
        self.initial_s_w = s_w
        self.current_s_w = s_w
        
    def set_cell_s_w(self, cell_index, s_w):
        """ Sets water saturation for cell_index. 
        """
        self.initial_s_w[cell_index] = s_w
        self.current_s_w[cell_index] = s_w
        
    def set_porosities(self, porosities):
        """ Sets cell porosities.
        """ 
        self.porosities = porosities

    def set_cell_porosity(self, cell_index, porosity):
        """ Sets porosity for cell_index. 
        """
        self.porosities[cell_index] = porosity

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


    def set_saturation_boundaries(self, boundary_marker, saturation_function):
        """ Sets the saturation boundaries for pressure 
        specified (Dirichlet) boundaries. 
        """
        self.saturation_boundaries[boundary_marker] = saturation_function

    def add_rate_well(self, injection_rate_water, injection_rate_oil, cell_index, well_name):
        """ Adds rate specified point source at the center 
        of cell_index. Returns the index of the new rate well. 
        Injection rate is in m^3/second
        """
        self.rate_wells.append(cell_index)
        self.rate_wells_rate_water.append(injection_rate_water)
        self.rate_wells_rate_oil.append(injection_rate_oil)
        self.rate_wells_name.append(well_name)
        self.pressure_files.append(open(well_name+".inj", 'w'))
        return len(self.rate_wells)-1

    def reset_wells(self):
        """ Removes all wells.
        """
        self.rate_wells = []
        self.rate_wells_rate_water = []
        self.rate_wells_rate_oil = []
        self.rate_wells_name = []
        
        self.pressure_wells = []
        self.pressure_wells_bhp = []
        self.pressure_wells_pi = []
        self.pressure_wells_name = []
        

    def add_pressure_well(self, bhp, PI, cell_index, well_name):
        """ Adds rate specified point source at the center 
        of cell_index. Returns the index of the new rate well. 
        Injection rate is in m^3/second
        """
        self.pressure_wells.append(cell_index)
        self.pressure_wells_bhp.append(bhp)
        self.pressure_wells_pi.append(PI)
        self.pressure_wells_name.append(well_name)
        self.production_files.append(open(well_name+".prod", 'w'))
        return len(self.pressure_wells)-1


    def get_well_rate_oil(self, well_index):
        """ Returns the oil production rate 
        for well_index
        """
        return self.rate_wells_rate_oil[well_index]

    def get_well_rate_water(self, well_index):
        """ Returns the water production rate 
        for well_index
        """
        return self.rate_wells_rate_water[well_index]

    def set_time_step_size(self, delta_t):
        """ Sets time step size. 
        """
        self.delta_t = delta_t

    def set_number_of_time_steps(self, number_of_time_steps):
        """ Set number of time steps taken. 
        """
        self.number_of_time_steps = number_of_time_steps
    

    def set_saturation_substeps(self, number_of_saturation_substeps):
        """ Sets the number of saturation steps to take between 
        regular pressure time step intervals. The default is 1, meaning 
        take one saturation step for each pressure step. 
        """
        self.saturation_time_steps = number_of_saturation_substeps

    def initialize_system(self):
        """ Constructs the initial matrices 
        used to construct the saddle-point 
        problem. 
        """
        #self.pressure_well_outputs = map(lambda n:open(n+".dat", 'w'), self.pressure_wells_name)

        self.mfd.set_mesh(self.mesh)
        [[div_data, div_row, div_col], 
         [div_t_data, div_t_row, div_t_col]] = self.mfd.build_div()
        print "building m"
        [self.m_x_coo_data, 
         m_x_coo_row, 
         m_x_coo_col] = self.mfd.build_m(save_update_info=True)
        print "done building m"

        print len(self.m_x_coo_data)

        self.div =  sparse.coo_matrix((np.array(div_data), 
                                       (np.add(np.array(div_row), 
                                               -self.mesh.get_number_of_faces()), 
                                        np.array(div_col))))
        self.div = self.div.tocsr()

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

        self.c_end = len(c_data) + self.c_start
        
        lhs_data += coupling_data
        lhs_row += coupling_row
        lhs_col += coupling_col

        # Convert m_x_coo_data to numpy array. 
        self.m_x_coo_data = np.array(self.m_x_coo_data)

        self.lhs_coo = sparse.coo_matrix((np.array(lhs_data), 
                                          (np.array(lhs_row), 
                                           np.array(lhs_col))))
        
        del lhs_data
        del lhs_row
        del lhs_col
        del c_data
        del c_row
        del c_col
        del div_data
        del div_row
        del div_col

        del div_t_data
        del div_t_row
        del div_t_col
        
        print "done building LHS"
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
        The two-phase system solves for p_o and s_w. Using IMPES, 
        we start by solving the pressure system using the update_
        pressure routine. After that, the saturations are updated 
        explicitly. 
        """
        self.mesh.output_vtk_mesh(self.model_name + "0", 
                                  [self.current_p_o, 
                                   self.mesh.get_cell_domain_all(), 
                                   range(self.mesh.get_number_of_cells()), 
                                   ], 
                                  ["pressure", "domain", "cell_number"])

        
        for time_step in range(1,self.number_of_time_steps+1):
            # update p_o and u_t (current pressure total flux) 
            self.time_step = time_step
            self.update_pressure()
            

            if time_step == 1 or time_step%10==0:
                self.find_upwinding_direction()
            for saturation_time_step in range(self.saturation_time_steps):
                #print "\t\t satuation step ", saturation_time_step
                self.update_saturation(time_step)
                
            
            if time_step%self.prod_output_frequency == 0:
                for (cell_index, output) in zip(self.rate_wells,                                                           
                                            self.pressure_files):

                    print >> output, time_step, self.current_p_o[cell_index], self.current_s_w[cell_index]

            
            if time_step%self.output_frequency == 0:



                self.mesh.output_vtk_mesh(self.model_name + str(time_step), 
                                          [self.current_s_w, self.current_p_o], 
                                          ["sw", "POIL"])
                #print "mimetic sum sw = ", sum(self.current_s_w)
                print "time step", time_step

                self.time_step_output(self.current_time, time_step)

            self.current_time = time_step*self.delta_t
            #print time_step
        
    def update_pressure(self):
        """ Solve the pressure system for p_o. self.current_p_o and 
        self.current_u_t are considered the pressure and velocity 
        at time n, and routine computes these quantities for time 
        n+1. 
        """
        # po_k, ut_k are the current newton iteration approximations 
        # to pressure and velocity.         
        po_k = np.array(self.current_p_o)
        ut_k = np.array(self.current_u_t)
        
        newton_residual = 100.

        newton_step = 0

        while abs(newton_residual > self.newton_threshold):
            current_total_mobility = self.water_mobility(self.current_s_w, po_k)
            current_total_mobility += self.oil_mobility(self.current_s_w, po_k)
            current_total_mobility = 1./current_total_mobility
            
            current_c_matrix =  self.ref_density_water*self.compressibility_water*self.current_s_w
            current_c_matrix += self.ref_density_oil*self.compressibility_oil*(1.-self.current_s_w)
            current_c_matrix *= self.porosities*self.mesh.cell_volume[:self.mesh.get_number_of_cells()]
            current_c_matrix /= self.delta_t
           
            self.mfd.update_m(self.lhs_coo.data[:self.m_x_coo_length], current_total_mobility)

            
            for (cell_index, pressure_pi) in zip(self.pressure_wells, 
                                                 self.pressure_wells_pi): 

                current_c_matrix[cell_index] += pressure_pi*1./current_total_mobility[cell_index]
                
            self.lhs_coo.data[self.c_start:self.c_end] = current_c_matrix

            lhs = self.lhs_coo.tocsr()

            ## J(x_n)(x_{n+1}-x_n) = -F(x_n)
            ## This line applies F(x_n)
            ut_k_po_k_combo = np.concatenate((ut_k, po_k))
            rhs = -self.mfd.build_rhs()
            rhs += lhs.dot(ut_k_po_k_combo)

            f2sum_l = np.ones(self.mesh.get_number_of_cells())
            f2sum_l *= self.ref_density_water*self.current_s_w
            f2sum_l *= self.porosities/self.delta_t
            f2sum_l *= self.mesh.cell_volume[:self.mesh.get_number_of_cells()]

            f2sum2_l = np.ones(self.mesh.get_number_of_cells())
            f2sum2_l *= self.ref_density_oil
            f2sum2_l *= 1.-self.current_s_w
            f2sum2_l *= self.porosities/self.delta_t
            f2sum2_l *= self.mesh.cell_volume[:self.mesh.get_number_of_cells()]

            f2sum3_l = np.zeros(self.mesh.get_number_of_cells())
            f2sum3_l += self.ref_density_water*(1.+self.compressibility_water*
                                                (self.current_p_o[cell_index]))
            f2sum3_l *= self.current_s_w
            f2sum3_l += self.ref_density_oil*(1+self.compressibility_oil*self.current_p_o)\
                *(1.-self.current_s_w)

            f2sum3_l *= self.porosities/self.delta_t
            f2sum3_l *= self.mesh.cell_volume[:self.mesh.get_number_of_cells()]

            rhs[self.mesh.get_number_of_faces():] += f2sum_l
            rhs[self.mesh.get_number_of_faces():] += f2sum2_l
            rhs[self.mesh.get_number_of_faces():] -= f2sum3_l

            for (well_index, cell_index) in enumerate(self.rate_wells):
                rhs[cell_index+self.mesh.get_number_of_faces()] += -self.get_well_rate_water(well_index)
                rhs[cell_index+self.mesh.get_number_of_faces()] += -self.get_well_rate_oil(well_index)

            for (cell_index, bhp, pressure_pi) in zip(self.pressure_wells, 
                                                      self.pressure_wells_bhp, 
                                                      self.pressure_wells_pi): 
                rhs[cell_index+self.mesh.get_number_of_faces()] -= \
                    pressure_pi*bhp*1./current_total_mobility[cell_index]
                
            newton_residual = np.linalg.norm(rhs)/np.linalg.norm(np.ones(len(rhs)))

            if newton_residual > self.newton_threshold:
                #print "solving..."
                newton_solution = dsolve.spsolve(lhs, -rhs)
                delta_po_k = newton_solution[self.mesh.get_number_of_faces():]
                delta_ut_k = newton_solution[:self.mesh.get_number_of_faces()]
                
                po_k += delta_po_k
                ut_k += delta_ut_k

            #print "\t\t", newton_step, newton_residual 
            newton_step += 1
            
        self.previous_p_o = np.array(self.current_p_o)
        self.previous_u_t = np.array(self.current_u_t)
        
        self.current_p_o = po_k
        self.current_u_t = ut_k
    
    def find_upwinding_direction(self):
        """Computes the upwinding direction 
        for each face in the mesh. 
        """
        self.upwinded_face_cell = []
        for cell_index in range(self.mesh.get_number_of_cells()):
            for [face_index, orientation] in zip(self.mesh.get_cell(cell_index), 
                                          self.mesh.get_cell_normal_orientation(cell_index)):
                if orientation*self.current_u_t[face_index]>0:
                    self.upwinded_face_cell.append([face_index, cell_index])

        ## Set fractional flow for dirichlet cells set by points
        ## based on up-winded flow direction. If the flow is out 
        ## of the cell, this is done automatically in previous 
        ## loops. However, for flow into the cell, we must find 
        ## the saturation from the cell it points to. 
        for face_index in self.mesh.get_dirichlet_pointer_faces():
            (cell_index, orientation) = self.mesh.get_dirichlet_pointer(face_index)
            if self.current_u_t[face_index]*orientation<0.:
                self.upwinded_face_cell.append([face_index, cell_index])

        self.current_f_w = np.zeros(self.mesh.get_number_of_faces())
        self.current_u_w = np.zeros(self.mesh.get_number_of_faces())

    def update_saturation(self, time_step):
        """ Updates water satuation based on the current p_o and u_t. 
        """
        sat_delta_t = self.delta_t/float(self.saturation_time_steps)

        water_mob = self.water_mobility(self.current_s_w, self.current_p_o)
        oil_mob = self.oil_mobility(self.current_s_w, self.current_p_o)

        for [face_index, cell_index] in self.upwinded_face_cell:
            self.current_f_w[face_index] = water_mob[cell_index]/(water_mob[cell_index]+oil_mob[cell_index])
       
        # Update saturation for faces on the boundary.         
        for boundary_marker in self.saturation_boundaries.keys():
            saturation_function = self.saturation_boundaries[boundary_marker]
            for (boundary_index, boundary_orientation) in\
                    self.mesh.get_boundary_faces_by_marker(boundary_marker):
                
                if boundary_orientation*self.current_u_t[boundary_index]<0:
                    quad_sum = 0.
                    for (quad_points, quad_weights) \
                            in zip(self.mesh.get_face_quadrature_points(boundary_index), 
                                   self.mesh.get_face_quadrature_weights(boundary_index)):
                            quad_sum += quad_weights*saturation_function(quad_points)

                    current_saturation = quad_sum/self.mesh.get_face_area(boundary_index)
                    current_pressure = self.mesh.get_dirichlet_boundary_value_by_face(boundary_index)
                    self.current_f_w[boundary_index]=self.water_mobility(current_saturation, 
                                                                    current_pressure)
                    self.current_f_w[boundary_index]/=(self.water_mobility(current_saturation, 
                                                                      current_pressure)+\
                                                      self.oil_mobility(current_saturation, 
                                                                        current_pressure))

            
        self.current_u_w[:] = self.current_f_w[:]
        np.multiply(self.current_u_w, self.current_u_t, self.current_u_w)

        div_uw = self.div.dot(self.current_u_w)

        # Aussming zero capillary pressure. 
        current_p_w = self.current_p_o
        previous_p_w = self.previous_p_o

        next_s_w = np.zeros(self.mesh.get_number_of_cells())
        
        for cell_index in range(self.mesh.get_number_of_cells()):
            new_s_w = (self.ref_density_water*(1+self.compressibility_water*
                                               previous_p_w[cell_index]))*self.current_s_w[cell_index]
            new_s_w /= (self.ref_density_water*(1.+self.compressibility_water * current_p_w[cell_index]))
            next_s_w[cell_index] += new_s_w  
          
            new_s_w = -sat_delta_t
            new_s_w /= self.porosities[cell_index]
            new_s_w /= self.ref_density_water
            new_s_w /= (1.+self.compressibility_water*current_p_w[cell_index])
            new_s_w /= self.mesh.get_cell_volume(cell_index)
            new_s_w *= div_uw[cell_index]

            next_s_w[cell_index] += new_s_w            

        for cell_index in self.rate_wells:
            well_index = self.rate_wells.index(cell_index)
            new_s_w = sat_delta_t
            
            new_s_w /= self.porosities[cell_index]
            new_s_w /= 1.+self.compressibility_water*current_p_w[cell_index]
            new_s_w /=self.ref_density_water

            new_s_w *= self.rate_wells_rate_water[well_index]/self.mesh.get_cell_volume(cell_index)
            
            next_s_w[cell_index] += new_s_w            

        # Update saturations for cells whose forcing function is computed 
        # from the flux of other faces. 
        for cell_index in self.mesh.get_forcing_pointer_cells():
            for (face_index, orientation) in self.mesh.get_forcing_pointers_for_cell(cell_index):
                new_s_w = self.current_u_w[face_index]
                new_s_w *= orientation*self.mesh.get_face_area(face_index)*sat_delta_t
                new_s_w /= self.porosities[cell_index]
                new_s_w /= self.ref_density_water
                new_s_w /= (1.+self.compressibility_water*current_p_w[cell_index])
                new_s_w /= self.mesh.get_cell_volume(cell_index)
                        
                next_s_w[cell_index] += new_s_w

        for (cell_index, bhp, pressure_pi, output) in zip(self.pressure_wells, 
                                                          self.pressure_wells_bhp, 
                                                          self.pressure_wells_pi, 
                                                          self.production_files):
            water_production = pressure_pi*(bhp-self.current_p_o[cell_index])*sat_delta_t
            water_production *= water_mob[cell_index]

            water_production /= self.porosities[cell_index]*self.ref_density_water*\
                (1.+self.compressibility_water*\
                     current_p_w[cell_index])
            water_production /= self.mesh.get_cell_volume(cell_index)

            oil_production = pressure_pi*(bhp-self.current_p_o[cell_index])*sat_delta_t
            oil_production *= oil_mob[cell_index]

            oil_production /= self.porosities[cell_index]*self.ref_density_oil*\
                (1.+self.compressibility_oil*\
                     self.current_p_o[cell_index])
            oil_production /= self.mesh.get_cell_volume(cell_index)
            
            if time_step%self.prod_output_frequency == 0:
                oil_production_modified = oil_production * (-1)
                water_production_modified = water_production * (-1)
                
                print >> output, time_step, oil_production_modified, water_production_modified

            next_s_w[cell_index] += water_production
            
        self.current_s_w[:] = next_s_w[:]


        
