from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from scipy import sparse, diag
from scipy import interpolate
import scipy.sparse.linalg.dsolve as dsolve
import scipy.sparse.linalg as linalg

import mimpy.mfd.mfd as mfd
from six.moves import range
from six.moves import zip

try:
    import petsc4py, sys
    petsc4py.init(sys.argv)
    from petsc4py import PETSc
    petsc4py_exists = True
except:
    petsc4py_exists = False

def get_python_matrix(shell):

    ## From Ethan Coon
    M = PETSc.Mat().create()
    M.setSizes(shell.size)
    M.setType('python')
    M.setPythonContext(shell)
    return M

class SchurComplementMat():
    def __init__(self, D, DT, M, C):
        """
        | M    DT |  |v|   |f1|
        |         |  | | = |  |
        | D    C  |  |p|   |f2|
        """
        self.D = D
        self.DT = DT
        self.M = M
        self.C = C
        self.size = C.size
        self.context = None

        self.ksp = PETSc.KSP().create()
        self.ksp.create(PETSc.COMM_WORLD)
        self.ksp.setType("cg")
        self.ksp.getPC().setType("jacobi")
        self.ksp.setOperators(self.M, self.M)

    def set_c(self, C):
        """ Sets matrix C.
        """
        self.C = C

    def create(self, M):
        pass

    def update_solver(self):
        self.ksp = PETSc.KSP().create()
        self.ksp.create(PETSc.COMM_WORLD)
        self.ksp.setType("cg")
        self.ksp.getPC().setType("jacobi")
        self.ksp.setOperators(self.M, self.M)

    def local_solve(self, dtx, m_inv_x):
        self.ksp.solve(dtx, m_inv_x)

    def mult(self, A, x, y):
        """ y = A x
        Applies (-DM^{-1}DT + C)x.

        Order of operations:
        dtx = DT x
        m_inv_x = M^{-1} dtx

        return -D m_inv_x + C x
        """
        dtx, m_inv_x = self.M.getVecs()

        self.DT.mult(x, dtx)

        self.local_solve(dtx, m_inv_x)

        self.D.mult(-m_inv_x, y)
        self.C.multAdd(x, y, y)

class SchurMatPC(object):
    def __init__(self, M):
        self.M = M

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

        self.newton_solution = None

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
        self.newton_step_max = 10
        # Sets how often to output
        # solution by time step number.
        self.output_frequency = 1
        self.prod_output_frequency = 1

        # Model Name
        self.model_name = "model"

        self.solver = 0

    def set_model_name(self, name):
        """ Sets model name used for output.
        """
        self.model_name = name

    def set_solver(self, tag):
        """ Sets the solution method.

        :param str tag: Indicates which solver to use:
            - "default" => uses spsolve from scipy.
            - "PETSc" => uses the petsc4py library

        :return: None
        """
        if tag == "PETSc":
            if petsc4py_exists:
                self.solver = 1
            else:
                raise Exception("petsc4py not installed")

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
        self.mfd.set_m_e_construction_method(6)

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

    def set_corey_relperm(self,
                          n_o,
                          n_w,
                          k_rw_o=1.):
        """ Sets a Corey-type relative permeability function.
        """

        def krw(se):
            return_array = np.zeros(len(se))
            return_array += se
            return_array[return_array < 0.] = 0.
            return_array[return_array > 1.] = 1.
            return_array = k_rw_o*(return_array**n_w)
            return return_array

        def kro(se):
            return_array = np.zeros(len(se))
            return_array += se
            return_array[return_array < 0.] = 0.
            return_array[return_array > 1.] = 1.
            return_array = (1.0-return_array)**n_o
            return return_array
        self.krw = krw
        self.kro = kro
    def set_kro(self, kro):
        """ Set oil relative permeability function.
        """
        self.kro = kro

    def set_krw(self, krw):
        """ Set water relative permeability function.
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
        """ Computes scaled saturation value
        se => [0, 1] from physical water saturation.
        """
        se = water_saturation-self.residual_saturation_water
        se /= 1.-self.residual_saturation_water-self.residual_saturation_oil

        return se

    def water_mobility(self, water_saturation, water_pressure):
        """ Returns water mobility
        """
        mobility = self.krw(self.sefromsw(water_saturation))
        mobility /= self.viscosity_water
        mobility *= self.ref_density_water
        mobility *= (1.+self.compressibility_water*water_pressure)

        return mobility

    def oil_mobility(self, water_saturation, oil_pressure):
        """ Returns oil mobility.
        """
        mobility = self.kro(self.sefromsw(water_saturation))
        mobility /= self.viscosity_oil
        mobility *= self.ref_density_oil
        mobility *= (1.+self.compressibility_oil*oil_pressure)

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
        raise Exception("deprecated use of intialize u_t")
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

    def add_rate_well(self,
                      injection_rate_water,
                      injection_rate_oil,
                      cell_index, well_name):
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
        self.mfd.set_mesh(self.mesh)
        [[div_data, div_row, div_col],
         [div_t_data, div_t_row, div_t_col]] = self.mfd.build_div()
        print("building m")
        [self.m_x_coo_data,
         m_x_coo_row,
         m_x_coo_col] = self.mfd.build_m(save_update_info=True)
        print("done building m")

        print(len(self.m_x_coo_data))

        self.current_u_t = np.zeros(self.mfd.flux_dof)
        self.initial_u_t = np.zeros(self.mfd.flux_dof)

        self.div = sparse.coo_matrix((np.array(div_data),
                                       (np.add(np.array(div_row),
                                               -self.mfd.flux_dof),
                                        np.array(div_col))))
        self.div = self.div.tocsr()

        self.m_x_coo_length = len(self.m_x_coo_data)

        # The data for the bottom right should be zeros.
        [c_data, c_row, c_col] = self.mfd.build_bottom_right()

        [coupling_data,
         coupling_row,
         coupling_col] = self.mfd.build_coupling_terms()

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

        # RHS construction is for Neumann and Dirichlet
        # boundaries specified by the mesh.
        self.rhs_mfd = self.mfd.build_rhs()

        self.newton_solution = np.zeros(self.mesh.get_number_of_cells()+
                                        self.mfd.flux_dof)

        if self.solver == 1:
            lhs_csr = self.lhs_coo.tocsr()

            self.lhs_petsc = PETSc.Mat()
            self.lhs_petsc.create(PETSc.COMM_WORLD)
            self.dense_ksp = PETSc.KSP().create()
            self.dense_ksp.create(PETSc.COMM_WORLD)
            self.dense_ksp.setOptionsPrefix("dense_")
            self.dense_ksp.setType("gmres")
            self.dense_ksp.getPC().setType("lu")
            self.dense_ksp.setFromOptions()

            m_info = self.mfd.build_m()
            [div_info, div_t_info] = self.mfd.build_div(0)
            c_info = self.mfd.build_bottom_right(0., shift=0)
            #Skipping coupling info
            self.m_coo = sparse.coo_matrix((m_info[0],
                                            (m_info[1], m_info[2])))

            m_csr = self.m_coo.tocsr()

            self.m_petsc = PETSc.Mat()
            self.m_petsc.create(PETSc.COMM_WORLD)
            self.m_petsc.createAIJWithArrays(size=m_csr.shape,
                                             csr=(m_csr.indptr,
                                                  m_csr.indices,
                                                  m_csr.data))

            self.m_petsc.assemblyBegin()
            self.m_petsc.assemblyEnd()

            self.div_coo = \
                sparse.coo_matrix((div_info[0],
                                   (div_info[1], div_info[2])),
                                  shape = (self.mesh.get_number_of_cells(),
                                           self.mfd.flux_dof))

            self.div_csr = self.div_coo.tocsr()
            self.div_petsc = PETSc.Mat()
            self.div_petsc.create(PETSc.COMM_WORLD)
            self.div_petsc.createAIJWithArrays(
                size=(self.mesh.get_number_of_cells(),
                      self.mfd.flux_dof),
                csr=(self.div_csr.indptr,
                     self.div_csr.indices,
                     self.div_csr.data))

            self.div_petsc.setUp()

            self.div_t_coo = \
                sparse.coo_matrix((div_t_info[0],
                                   (div_t_info[1], div_t_info[2])),
                                  shape = (self.mfd.flux_dof,
                                           self.mesh.get_number_of_cells()))

            self.div_t_csr = self.div_t_coo.tocsr()

            self.div_t_petsc = PETSc.Mat()
            self.div_t_petsc.create(PETSc.COMM_WORLD)
            self.div_t_petsc.createAIJWithArrays(
                size=(self.mfd.flux_dof,
                      self.mesh.get_number_of_cells()),
                csr=(self.div_t_csr.indptr,
                     self.div_t_csr.indices,
                     self.div_t_csr.data))

            self.div_t_petsc.setUp()

            self.c_coo = \
                sparse.coo_matrix((c_info[0],
                                   (c_info[1], c_info[2])),
                                  shape = (self.mesh.get_number_of_cells(),
                                           self.mesh.get_number_of_cells()))

            c_csr = self.c_coo.tocsr()
            self.c_petsc = PETSc.Mat()
            self.c_petsc.create(PETSc.COMM_WORLD)
            self.c_petsc.createAIJWithArrays(
                size=(self.mesh.get_number_of_cells(),
                      self.mesh.get_number_of_cells()),
                csr=(c_csr.indptr,
                     c_csr.indices,
                     c_csr.data))

            self.c_petsc.setUp()

            self.c_petsc.assemblyBegin()
            self.c_petsc.assemblyEnd()

            self.div_petsc.assemblyBegin()
            self.div_petsc.assemblyEnd()

            self.div_t_petsc.assemblyBegin()
            self.div_t_petsc.assemblyEnd()

            self.schur_mat = SchurComplementMat(self.div_petsc,
                                                self.div_t_petsc,
                                                self.m_petsc,
                                                self.c_petsc)

            self.schur_petsc = get_python_matrix(self.schur_mat)
            self.schur_petsc.setUp()

            m_diag = m_csr.diagonal()
            m_diag = 1./m_diag
            m_diag = sparse.csr_matrix((m_diag,
                                        (list(range(self.mfd.flux_dof)),
                                         list(range(self.mfd.flux_dof)))))

            self.last_solution = np.zeros(self.mesh.get_number_of_cells())

            pc_matrix = -self.div_csr.dot(m_diag.dot(self.div_t_csr))

            pc_matrix.sort_indices()

            self.pc_petsc = PETSc.Mat()
            self.pc_petsc.create(PETSc.COMM_WORLD)
            self.pc_petsc.createAIJWithArrays(
                size=(self.mesh.get_number_of_cells(),
                      self.mesh.get_number_of_cells()),
                csr=(pc_matrix.indptr,
                     pc_matrix.indices,
                     pc_matrix.data))
            self.pc_petsc.assemblyBegin()
            self.pc_petsc.assemblyEnd()

            self.ksp = PETSc.KSP()
            self.ksp.create(PETSc.COMM_WORLD)
            self.ksp.setType("cg")
            self.ksp.getPC().setType("bjacobi")
            self.ksp.setFromOptions()
            return

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
                                   list(range(self.mesh.get_number_of_cells()))],
                                  ["pressure", "domain", "cell_number"])

        for time_step in range(1, self.number_of_time_steps + 1):
            # update p_o and u_t (current pressure total flux)
            self.time_step = time_step
            self.update_pressure(time_step)

            if time_step == 1 or time_step % 10 == 0:
                self.find_upwinding_direction()
            for saturation_time_step in range(self.saturation_time_steps):
                self.update_saturation(time_step)

            if time_step % self.prod_output_frequency == 0:
                for (cell_index, output) in zip(self.rate_wells,
                                            self.pressure_files):
                    print(time_step, self.current_p_o[cell_index], end=' ', file=output)
                    print(self.current_s_w[cell_index], file=output)

            if time_step % self.output_frequency == 0:
                self.mesh.output_vtk_mesh(self.model_name + str(time_step),
                                          [self.current_s_w, 
                                           self.current_p_o,
                                           self.mesh.get_cell_domain_all()],
                                          ["sw", "POIL", "domain"])
                print("time step", time_step)

                self.time_step_output(self.current_time, time_step)

            self.current_time = time_step*self.delta_t

    def update_pressure(self, time_step):
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

            current_c_matrix = self.ref_density_water*self.current_s_w
            current_c_matrix *= self.compressibility_water

            current_c_matrix += self.ref_density_oil*(self.compressibility_oil
                                                      *(1.-self.current_s_w))

            current_c_matrix *= self.porosities
            current_c_matrix *= \
                self.mesh.cell_volume[:self.mesh.get_number_of_cells()]
            current_c_matrix /= self.delta_t

            self.mfd.update_m(self.lhs_coo.data[:self.m_x_coo_length],
                              current_total_mobility)

            for (cell_index, pressure_pi) in zip(self.pressure_wells,
                                                 self.pressure_wells_pi):
                current_c_matrix[cell_index] += \
                    pressure_pi*1./current_total_mobility[cell_index]

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
                                                (self.current_p_o))
            f2sum3_l *= self.current_s_w
            f2sum3_l += self.ref_density_oil*\
                (1+self.compressibility_oil*self.current_p_o)*\
                (1.-self.current_s_w)

            f2sum3_l *= self.porosities/self.delta_t
            f2sum3_l *= self.mesh.cell_volume[:self.mesh.get_number_of_cells()]

            rhs[self.mfd.flux_dof:] += f2sum_l
            rhs[self.mfd.flux_dof:] += f2sum2_l
            rhs[self.mfd.flux_dof:] -= f2sum3_l

            for (well_index, cell_index) in enumerate(self.rate_wells):
                rhs[cell_index+self.mfd.flux_dof] += \
                    -self.get_well_rate_water(well_index)
                rhs[cell_index+self.mfd.flux_dof] += \
                    -self.get_well_rate_oil(well_index)

            for (cell_index, bhp, pressure_pi) in zip(self.pressure_wells,
                                                      self.pressure_wells_bhp,
                                                      self.pressure_wells_pi):
                rhs[cell_index+self.mfd.flux_dof] -= \
                    pressure_pi*bhp*1./current_total_mobility[cell_index]

            newton_residual = np.linalg.norm(rhs)/float(len(rhs))

            if newton_residual > self.newton_threshold:
                if self.solver == 0:
                    self.newton_solution = dsolve.spsolve(lhs, -rhs)
                    delta_po_k = self.newton_solution[self.mfd.flux_dof:]
                    delta_ut_k = self.newton_solution[:self.mfd.flux_dof]

                if self.solver == 1:
                    self.mfd.update_m(self.m_coo.data, current_total_mobility)
                    m_csr = self.m_coo.tocsr()
                    self.m_petsc.createAIJWithArrays(size=m_csr.shape,
                                                     csr=(m_csr.indptr,
                                                          m_csr.indices,
                                                          m_csr.data))
                    self.m_petsc.setUp()
                    self.m_petsc.assemblyBegin()
                    self.m_petsc.assemblyEnd()

                    self.c_coo.data = current_c_matrix
                    c_csr = self.c_coo.tocsr()
                    self.c_petsc.createAIJWithArrays(
                        size=(self.mesh.get_number_of_cells(),
                              self.mesh.get_number_of_cells()),
                        csr=(c_csr.indptr,
                             c_csr.indices,
                             c_csr.data))

                    self.c_petsc.setUp()
                    self.c_petsc.assemblyBegin()
                    self.c_petsc.assemblyEnd()

                    m_diag = m_csr.diagonal()
                    m_diag = 1./m_diag
                    m_diag = sparse.csr_matrix((m_diag,
                                                (list(range(self.mfd.flux_dof)),
                                                 list(range(self.mfd.flux_dof)))))

                    pc_matrix = -self.div_csr.dot(m_diag.dot(self.div_t_csr))
                    pc_matrix += c_csr
                    pc_matrix.sort_indices()

                    self.pc_petsc = PETSc.Mat()
                    self.pc_petsc.create(PETSc.COMM_WORLD)
                    self.pc_petsc.createAIJWithArrays(
                        size=(self.mesh.get_number_of_cells(),
                              self.mesh.get_number_of_cells()),
                        csr=(pc_matrix.indptr,
                             pc_matrix.indices,
                             pc_matrix.data))

                    self.pc_petsc.assemblyBegin()
                    self.pc_petsc.assemblyEnd()

                    self.schur_mat.set_c(self.c_petsc)

                    self.schur_mat.update_solver()

                    x, y = self.c_petsc.getVecs()
                    df1, f1 = self.m_petsc.getVecs()

                    f1.setArray(rhs[:self.mfd.flux_dof])
                    self.schur_mat.ksp.solve(f1, df1)

                    df1 = self.div_coo.dot(df1)
                    temp1, temp2 = self.c_petsc.getVecs()
                    temp1.setArray(np.ones(self.mesh.get_number_of_cells()))
                    self.schur_mat.mult(None, temp1, temp2)

                    x.setArray(df1-rhs[self.mfd.flux_dof:])
                    self.ksp.setOperators(self.schur_petsc, self.pc_petsc)
                    self.ksp.solve(x, y)
                    if newton_step == 1:
                        self.last_solution = np.array(y.getArray())

                    delta_po_k = y
                    f1_minvp, delta_ut_k = self.m_petsc.getVecs()
                    f1_minvp.setArray(-rhs[:self.mfd.flux_dof]-
                                       self.div_t_coo.dot(y.getArray()))
                    self.schur_mat.ksp.solve(f1_minvp, delta_ut_k)

                    delta_po_k = delta_po_k.getArray()
                    delta_ut_k = delta_ut_k.getArray()

                po_k += delta_po_k
                ut_k += delta_ut_k

            print("\t\t", newton_step, newton_residual)
            newton_step += 1
            if newton_step > self.newton_step_max:
                1/0

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
            current_cell = self.mesh.get_cell(cell_index)
            current_orientation = \
                self.mesh.get_cell_normal_orientation(cell_index)
            face_orientation_list = list(zip(current_cell, current_orientation))
            for [face_index, orientation] in face_orientation_list:
                current_direction = orientation
                flux_index = self.mfd.face_to_flux[face_index, 0]
                current_direction *= self.current_u_t[flux_index]
                if current_direction > 0:
                    self.upwinded_face_cell.append([flux_index, cell_index])

        ## Set fractional flow for dirichlet cells set by points
        ## based on up-winded flow direction. If the flow is out
        ## of the cell, this is done automatically in previous
        ## loops. However, for flow into the cell, we must find
        ## the saturation from the cell it points to.
        for face_index in self.mesh.get_dirichlet_pointer_faces():
            (cell_index, orientation) = \
                self.mesh.get_dirichlet_pointer(face_index)
            flux_index = self.mfd.face_to_flux[face_index]
            if self.current_u_t[flux_index]*orientation<0.:
                self.upwinded_face_cell.append([flux_index, cell_index])

        self.current_f_w = np.zeros(self.mfd.flux_dof)
        self.current_u_w = np.zeros(self.mfd.flux_dof)

    def update_saturation(self, time_step):
        """ Updates water satuation based on the current p_o and u_t.
        """
        sat_delta_t = self.delta_t / float(self.saturation_time_steps)

        water_mob = self.water_mobility(self.current_s_w, self.current_p_o)
        oil_mob = self.oil_mobility(self.current_s_w, self.current_p_o)

        for [face_index, cell_index] in self.upwinded_face_cell:
            self.current_f_w[face_index] = water_mob[cell_index]
            self.current_f_w[face_index] /= (water_mob[cell_index]+
                                             oil_mob[cell_index])

        # Update saturation for faces on the boundary.
        for boundary_marker in list(self.saturation_boundaries.keys()):
            saturation_function = self.saturation_boundaries[boundary_marker]
            for (boundary_index, boundary_orientation) in\
                    self.mesh.get_boundary_faces_by_marker(boundary_marker):
                flux_index = self.mfd.face_to_flux[boundary_index]
                if boundary_orientation * self.current_u_t[flux_index] < 0:
                    face_centroid = \
                        self.mesh.get_face_real_centroid(boundary_index)
                    current_saturation = saturation_function(face_centroid)
                    current_saturation = np.array([current_saturation])
                    current_pressure = \
                        np.array([self.mfd.get_dirichlet_value(boundary_index)])
                    self.current_f_w[flux_index] = \
                        self.water_mobility(current_saturation,
                                            current_pressure)

                    self.current_f_w[flux_index] /= \
                        (self.water_mobility(current_saturation,
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

        new_s_w_l = previous_p_w*self.compressibility_water+1.
        new_s_w_l *= self.ref_density_water
        new_s_w_l *= self.current_s_w
        new_s_w_l /= self.ref_density_water
        new_s_w_l /= 1.+self.compressibility_water*current_p_w

        next_s_w += new_s_w_l

        new_s_w_l = 1./self.porosities
        new_s_w_l /= self.ref_density_water
        new_s_w_l /= (1.+self.compressibility_water*current_p_w)
        new_s_w_l /= self.mesh.cell_volume[:self.mesh.get_number_of_cells()]
        new_s_w_l *= div_uw[:self.mesh.get_number_of_cells()]
        new_s_w_l *= -sat_delta_t

        next_s_w += new_s_w_l

        for cell_index in self.rate_wells:
            well_index = self.rate_wells.index(cell_index)
            new_s_w = sat_delta_t

            new_s_w /= self.porosities[cell_index]
            new_s_w /= 1.+self.compressibility_water*current_p_w[cell_index]
            new_s_w /= self.ref_density_water

            new_s_w *= self.rate_wells_rate_water[well_index]
            new_s_w /= self.mesh.get_cell_volume(cell_index)

            next_s_w[cell_index] += new_s_w

        # Update saturations for cells whose forcing function is computed
        # from the flux of other faces.
        for cell_index in self.mesh.get_forcing_pointer_cells():
            for (face_index, orientation) in \
                    self.mesh.get_forcing_pointers_for_cell(cell_index):
                flux_index = self.mfd.face_to_flux[face_index]
                new_s_w = self.current_u_w[flux_index]
                new_s_w *= orientation
                new_s_w *= self.mesh.get_face_area(face_index)*sat_delta_t
                new_s_w /= self.porosities[cell_index]
                new_s_w /= self.ref_density_water
                new_s_w /= (1.+self.compressibility_water*
                            current_p_w[cell_index])
                new_s_w /= self.mesh.get_cell_volume(cell_index)
                next_s_w[cell_index] += new_s_w

        for well_number in range(len(self.pressure_wells)):
            cell_index = self.pressure_wells[well_number]
            bhp = self.pressure_wells_bhp[well_number]
            pressure_pi = self.pressure_wells_pi[well_number]
            output = self.production_files[well_number]

            water_production = pressure_pi
            water_production *= (bhp-self.current_p_o[cell_index])
            water_production *= sat_delta_t
            water_production *= water_mob[cell_index]

            water_production /= self.porosities[cell_index]
            water_production /= self.ref_density_water
            water_production /= (1.+self.compressibility_water*\
                                     current_p_w[cell_index])

            water_production /= self.mesh.get_cell_volume(cell_index)

            oil_production = pressure_pi
            oil_production *= (bhp-self.current_p_o[cell_index])
            oil_production *= sat_delta_t
            oil_production *= oil_mob[cell_index]

            oil_production /= self.porosities[cell_index]
            oil_production /= self.ref_density_oil
            oil_production /= (1.+self.compressibility_oil *\
                                   self.current_p_o[cell_index])

            oil_production /= self.mesh.get_cell_volume(cell_index)

            if time_step % self.prod_output_frequency == 0:
                oil_production_modified = oil_production*(-1)
                water_production_modified = water_production*(-1)

                print(time_step, oil_production_modified, end=' ', file=output)
                print(water_production_modified, file=output)

            next_s_w[cell_index] += water_production

        self.current_s_w[:] = next_s_w[:]



