
import numpy as np
import sys
import copy
import array
import mimpy.mfd.mfd_cython as mfd_cython

from scipy import sparse, diag
import scipy.sparse.linalg.dsolve as dsolve
import scipy.sparse.linalg as linalg
from multiprocessing import Pool

try:
    from petsc4py import PETSc
except:
    pass

def sign(x):
    """ Returns the sign of float x.
    """
    if abs(x) == x:
        return 1.
    else:
        return -1.

class MFD():
    """ The MFD class provides functionality for
    constructing mimetic discretization matrices
    for sovling elliptic PDE. An instance of MFD
    constructs the matrices based on the information
    it finds in an instance of the Mesh class.
    """
    def __init__(self):
        self.mesh = None
        self.lhs = None
        self.rhs = None
        self.solution = None

        self.check_m_e = False

        self.m_elist = []
        self.is_diag_list = []

        self.print_progress = False

        self.compute_diagonality = False

        # List indicating orthogonality of cells.
        self.diagonality_index_list = None

        self.m_e_construction_method = 0

        # Data needed for updating matrix Mx
        self.m_data_for_update = None
        self.m_e_locations = None

        # dict: {face_index: Dirichlet value, ... }
        self.dirichlet_boundary_values = {}

        # dict: {face_index: Neumann value, ... }
        self.neumann_boundary_values = {}

        # list: [F1, F2, F3, ....]
        self.cell_forcing_function = np.empty(shape=(0), dtype=float)

        # Dict from cell to a list of faces
        # in the cell that are also neumann faces.
        # Used for incroporating the boundary
        # conditions when building m and div_t.
        self.cell_faces_neumann = {}

        self.non_ortho_count = 0
        self.all_ortho = True

    def set_mesh(self, mesh):
        """ Links the MFD object to an instance of a Mesh class.
        """
        self.mesh = mesh
        self.cell_forcing_function.resize(self.mesh.get_number_of_cells(),
                                          refcheck=False)
        self.process_internal_boundaries()

    def set_compute_diagonality(self, setting):
        """ During matrix M_x construction, compute how
        diagonal the local stiffness matrices
        m_e. The equation is:

        ||M_E - diag(diag(M_E)||_2/||M_E||_2
        """
        self.compute_diagonality = setting

    def get_diagonality(self):
        """ Returns a list of cell diagonality indices.
        """
        return self.diagonality_index_list

    def set_m_e_construction_method(self, method_index):
        """ Set construction method for the local matrices
        m_e. Method indices correspond to:

        0 => U = R^TR(R^TN)^-1

        1 => U = trace(k^(-1))|E|
        """
        self.m_e_construction_method = method_index

    def build_div(self, shift=1):
        """ Build the div and -div^T matrices and returns
        arrays for constructing a coo sparse matrix
        representation. The lists returned are:

        [[div_data, div_row, div_col],
        [div_t_data, div_t_row, div_t_col]]
        """
        div_data = array.array('f')
        div_row = array.array('i')
        div_col = array.array('i')

        div_t_data = array.array('f')
        div_t_row = array.array('i')
        div_t_col = array.array('i')

        neumaan_boundary_indices = map(lambda x: x[0],
                                       self.get_neumann_boundary_values())
        for current_cell_index in range(self.mesh.get_number_of_cells()):
            current_cell = self.mesh.get_cell(current_cell_index)
            current_cell_orientations = \
                self.mesh.get_cell_normal_orientation(current_cell_index)

            neumann_faces = self.get_cell_faces_neumann(current_cell_index)

            for [face_index, face_orientation] in \
                    zip(current_cell, current_cell_orientations):

                new_entry = face_orientation*self.mesh.get_face_area(face_index)

                div_data.append(new_entry)
                div_row.append(current_cell_index+
                               shift*self.mesh.get_number_of_faces())
                div_col.append(face_index)

                if face_index not in neumann_faces:
                    div_t_data.append(-new_entry)
                    div_t_row.append(face_index)
                    div_t_col.append(current_cell_index+
                                     shift*self.mesh.get_number_of_faces())

        return [[div_data, div_row, div_col],
                [div_t_data, div_t_row, div_t_col]]

    def build_r_e(self, cell_index):
        """ Build matrix R_E for construction of
        local stiffness matrix M_E.
        The function defaults to using the
        real centroids of the cells and faces
        unless otherwise specified by functions
        is_using_face_shifted_centroid and
        is_using_cell_shifted_centroid.
        """
        r_e = np.zeros((self.mesh.get_number_of_cell_faces(cell_index),
                           self.mesh.dim))

        counter = 0

        if self.mesh.is_using_cell_shifted_centroid():
            cell_centroid = self.mesh.get_cell_shifted_centroid(cell_index)
        else:
            cell_centroid = self.mesh.get_cell_real_centroid(cell_index)
        if self.mesh.dim == 3:
            for face_index in self.mesh.get_cell(cell_index):
                if self.mesh.is_using_face_shifted_centroid():
                    face_centroid =\
                        self.mesh.get_face_shifted_centroid(face_index)
                else:
                    face_centroid = self.mesh.get_face_real_centroid(face_index)

                r_e[counter][0] = self.mesh.get_face_area(face_index)
                r_e[counter][0] *= (face_centroid[0]-cell_centroid[0])

                r_e[counter][1] = self.mesh.get_face_area(face_index)
                r_e[counter][1] *= (face_centroid[1]-cell_centroid[1])

                r_e[counter][2] = self.mesh.get_face_area(face_index)
                r_e[counter][2] *= (face_centroid[2]-cell_centroid[2])

                counter += 1

        return r_e

    def build_n_e(self, cell_index, k_unity = False):
        """ Build matrix N_E using the standard
        definition for the mimetic method.
        """
        number_of_faces = len(self.mesh.get_cell(cell_index))
        n_e = np.zeros((number_of_faces, self.mesh.dim))

        counter = 0

        face_face_orientation = \
            zip(self.mesh.get_cell(cell_index),
                self.mesh.get_cell_normal_orientation(cell_index))

        if k_unity:
            for [face_index, face_orientation] in face_face_orientation:
                n_e[counter, :] = self.mesh.get_face_normal(face_index)
                n_e[counter, :] *= face_orientation
                counter += 1
        else:
            for [face_index, face_orientation] in face_face_orientation:
                n_e[counter, :] = np.dot(self.mesh.get_cell_k(cell_index),
                                         self.mesh.get_face_normal(face_index))
                n_e[counter, :] *= face_orientation
                counter += 1

        return n_e

    def build_c_e(self, n_e):
        """ Construct matrix C_E. The matrix
        must satisfy N_E^T C_E = 0.
        """
        [u, s, v] = np.linalg.svd(np.transpose(n_e))

        if self.mesh.dim == 3:
            c_e = np.transpose(v)[:, 3:]

        if self.mesh.dim == 2:
            c_e = np.transpose(v)[:, 2:]

        return c_e

    def build_d_e(self, n_e):
        """ Construct matrix D_E. The matrix
        must satisfy R_E^T D_E = 0.
        """
        [u, s, v] = np.linalg.svd(np.transpose(n_e))

        if self.mesh.dim == 3:
            c_e = np.transpose(v)[:, 3:]

        if self.mesh.dim == 2:
            c_e = np.transpose(v)[:, 2:]

        return c_e

    def build_w_e(self, cell_index):
        """ Construct matrix W_E from Brezzi et al.
        paper.
        """
        r_e = self.build_r_e(cell_index)
        n_e = self.build_n_e(cell_index)

        d_e = self.build_d_e(r_e)
        current_k = self.mesh.get_cell_k(cell_index)
        k_inv = np.linalg.inv(current_k)

        w_e = n_e.dot(np.linalg.inv(n_e.T.dot(r_e)).dot(n_e.T))

        w_e += d_e.dot(d_e.T)
        w_e *= np.trace(current_k)
        w_e /= self.mesh.get_cell_volume(cell_index)

        return w_e

    def build_m_e(self, cell_index, k_unity = False):
        """ Construct the local stiffness matrix M_E.
        The construction is based on the relation:
        M_E = R_E (R_E^T N_E)^-1 R_E^T + C_E U C_E^T
        Matrix U is a parameter that can be chosen
        during construction. The options for setting
        parameter U are set using function
        set_m_e_construction_method.
        """
        r_e = self.build_r_e(cell_index)
        if k_unity:
            n_e = self.build_n_e(cell_index, k_unity)
            current_k = np.eye(self.mesh.dim)
        else:
            n_e = self.build_n_e(cell_index)
            current_k = self.mesh.get_cell_k(cell_index)
        is_ortho = True
        c_e = self.build_c_e(n_e)
        
        m_0 =  r_e.dot(np.linalg.inv(np.dot(r_e.T, n_e)).dot(r_e.T))

        if self.m_e_construction_method == 0:
            u = np.linalg.inv(np.dot(r_e.T, n_e))
            u = np.dot(np.dot(r_e.T, r_e), u)
            u = u[0, 0]
            m_e = m_0
            m_e += np.dot(c_e, np.dot(u, np.transpose(c_e)))

        elif self.m_e_construction_method == 1:
            u = self.mesh.get_cell_volume(cell_index)/np.trace(current_k)
            m_e = m_0
            m_e += np.dot(c_e, np.dot(u, np.transpose(c_e)))

        elif self.m_e_construction_method == 2:
            ## Based on producing RT0 for triangles
            ## from "The Mimetic Finite Difference Method"
            ## By Gianmarco Manzini,  Eqn 54.
            k_inv = np.linalg.inv(current_k)
            u = 0.
            for face_index in self.mesh.get_cell(cell_index):
                x_e_min_x_E = self.mesh.get_face_real_centroid(face_index)- \
                    self.mesh.get_cell_real_centroid(cell_index)
                u += np.dot(x_e_min_x_E.T, np.dot(k_inv, x_e_min_x_E))
            u /= self.mesh.dim**2
            u /= self.mesh.get_number_of_cell_faces(cell_index)
            m_e = m_0
            m_e += np.dot(c_e, np.dot(u, c_e.T))

        elif self.m_e_construction_method == 3:
            ## Based on Mimetic Finite Difference method
            ## in JCP 2013.
            lambda_c = np.linalg.inv(r_e.T.dot(n_e))
            lambda_c = lambda_c.dot(r_e.T)
            lambda_c = r_e.dot(lambda_c)
            lambda_c = np.trace(lambda_c)
            lambda_c *= .5

            m_e = np.eye(self.mesh.get_number_of_cell_faces(cell_index))
            m_e -= (n_e.dot(np.linalg.inv(n_e.T.dot(n_e)))).dot(n_e.T)
            m_e = lambda_c*(m_e)
            m_e += m_0

        elif self.m_e_construction_method == 4:
            m_e = np.linalg.inv(self.build_w_e(cell_index))

        elif self.m_e_construction_method == 5:
            diagonal = []
            def absmaxindex(vector):
                current_max = abs(vector[0])
                max_index = 0
                for i in range(1, len(vector)):
                    if abs(vector[i])>current_max:
                        max_index = i
                        current_max = abs(vector[i])
                return max_index

            for i in range(len(self.mesh.get_cell(cell_index))):
                max_index = absmaxindex(r_e[i, :])
                diagonal.append(r_e[i, max_index]/n_e[i, max_index])

            m_e = np.diag(diagonal)

        elif self.m_e_construction_method == 6:
            m_e =  r_e.dot(np.linalg.inv(np.dot(r_e.T, n_e)).dot(r_e.T))
            diagonal = []        
            for (face_index, orientation) in \
                    zip(self.mesh.get_cell(cell_index), 
                        self.mesh.get_cell_normal_orientation(cell_index)):
                    normal = orientation*self.mesh.get_face_normal(face_index)
                    entry = normal.dot(np.linalg.inv(current_k).dot(normal))
                    entry *= self.mesh.get_face_area(face_index)
                    entry *= np.linalg.norm(self.mesh.get_face_real_centroid(face_index)-
                                            self.mesh.get_cell_real_centroid(cell_index))
                    diagonal.append(entry)

            diagonal=  np.diag(diagonal)
            m_e += diagonal.dot(c_e.dot(c_e.T))

        if self.check_m_e:
            if np.linalg.norm(np.dot(m_e, n_e)-r_e) > 1.e-6:
                print "M_E N ne R", np.linalg.norm(np.dot(m_e, n_e)-r_e)

        if self.compute_diagonality:
            self.diagonality_index_list[cell_index] = \
                np.linalg.norm(m_e-np.diag(np.diag(m_e)))/np.linalg.norm(m_e)
            
            if self.diagonality_index_list[cell_index] > 1.e-8:
                self.all_ortho = False
        return m_e

    def build_m(self, save_update_info = False, k_unity = False):
        """ Construct the global matrix M_x. This is
        done by first constructing local matrices
        M_E and constructing a coo matrix for
        the global matrix.
        """
        if self.compute_diagonality:
            self.diagonality_index_list = \
                np.zeros(self.mesh.get_number_of_cells())

        m_data = array.array('f')
        m_row = array.array('i')
        m_col = array.array('i')

        if save_update_info:
            self.m_e_locations = [0]

        current_length = 0
        neumann_boundary_indices = \
            map(lambda x: x[0], self.get_neumann_boundary_values())

        total_work = self.mesh.get_number_of_cells()
        percentage_inc = 10.
        last_percent = 10.
        for cell_index in range(self.mesh.get_number_of_cells()):
            m_e = self.build_m_e(cell_index, k_unity)

            neumann_faces = self.get_cell_faces_neumann(cell_index)

            current_cell = self.mesh.get_cell(cell_index)
            current_orientation = \
                self.mesh.get_cell_normal_orientation(cell_index)
            for i in range(len(m_e)):
                global_i = current_cell[i]

                if global_i not in  neumann_faces:
                    for j in range(len(m_e)):
                        global_j = current_cell[j]

                        if abs(m_e[i][j]) > 1.e-40:
                            current_length += 1
                            m_data.append(m_e[i, j]*
                                          current_orientation[i]*
                                          current_orientation[j])
                            m_row.append(global_i)
                            m_col.append(global_j)

            if save_update_info:
                self.m_e_locations.append(current_length)

        for face_index in map(lambda x: x[0],
                                self.get_neumann_boundary_values()):
            m_data.append(1.)
            m_row.append(face_index)
            m_col.append(face_index)

        if save_update_info:
            self.m_data_for_update = np.array(m_data)
            self.m_e_locations = np.array(self.m_e_locations)

        return [m_data, m_row, m_col]


    def build_m_parallel(self, save_update_info = False, k_unity = False):
        """ Construct the global matrix M_x. This is
        done by first constructing local matrices
        M_E and constructing a coo matrix for
        the global matrix.
        """
        if self.compute_diagonality:
            self.diagonality_index_list = \
                np.zeros(self.mesh.get_number_of_cells())

        m_data = array.array('f')
        m_row = array.array('i')
        m_col = array.array('i')

        if save_update_info:
            self.m_e_locations = [0]

        current_length = 0
        neumann_boundary_indices = \
            map(lambda x: x[0], self.get_neumann_boundary_values())

        for cell_index in range(self.mesh.get_number_of_cells()):
            m_e = self.build_m_e(cell_index, k_unity)

            current_cell = self.mesh.get_cell(cell_index)
            current_orientation = \
                self.mesh.get_cell_normal_orientation(cell_index)
            for i in range(len(m_e)):
                global_i = current_cell[i]

                if global_i not in neumann_boundary_indices:
                    for j in range(len(m_e)):
                        global_j = current_cell[j]

                        if abs(m_e[i][j]) > 1.e-40:
                            current_length += 1
                            m_data.append(m_e[i, j]*
                                          current_orientation[i]*
                                          current_orientation[j])
                            m_row.append(global_i)
                            m_col.append(global_j)

            if save_update_info:
                self.m_e_locations.append(current_length)


        for face_index in map(lambda x: x[0],
                                self.get_neumann_boundary_values()):
            m_data.append(1.)
            m_row.append(face_index)
            m_col.append(face_index)

        print "All Ortho = ", self.all_ortho


        if save_update_info:
            self.m_data_for_update = np.array(m_data)

        return [m_data, m_row, m_col]


    def update_m(self, m_coo, multipliers):
        """ Updates matrix Mx using a list of constants
        that represent values multiplied by K.
        For this function to work, the
        save_update_info would have to be set
        to True while running build_m function.
        The MFD instance saves the original m_e
        matrices used to construct the full Mx
        matrix in coo format.
        """
        mfd_cython.update_m_fast(m_coo, 
                                 multipliers, 
                                 self.m_data_for_update,
                                 self.m_e_locations, 
                                 self.mesh.get_number_of_cells())

    def build_bottom_right(self, alpha = 0.):
        """ Build the matrix C in the global
        saddle point problem (bottom right
        side of the matrix). Returns
        data, row and col data for coo matrix.
        """
        bottom_right_data = array.array('f')
        bottom_right_row = array.array('i')
        bottom_right_col = array.array('i')

        if self.mesh.is_using_alpha_list:
            for cell_index in range(self.mesh.get_number_of_cells()):
                entry = (self.mesh.get_alpha_by_cell(cell_index)*
                         self.mesh.get_cell_volume(cell_index))
                bottom_right_data.append(entry)
                bottom_right_row.append(cell_index +
                                        self.mesh.get_number_of_faces())
                bottom_right_col.append(cell_index +
                                        self.mesh.get_number_of_faces())

        else:
            for cell_index in range(self.mesh.get_number_of_cells()):
                entry = alpha * self.mesh.get_cell_volume(cell_index)
                bottom_right_data.append(entry)
                bottom_right_row.append(cell_index +
                                        self.mesh.get_number_of_faces())
                bottom_right_col.append(cell_index +
                                        self.mesh.get_number_of_faces())

        return [bottom_right_data, bottom_right_row, bottom_right_col]

    def build_coupling_terms(self):
        """ Builds matrices L and L^T that
        couple the fracture domains
        with the reservoir as well as
        different reservoir domains.
        Used information found in the
        boundary and forcing function
        pointers.
        """
        coupling_data = array.array('f')
        coupling_row = array.array('i')
        coupling_col = array.array('i')

        for cell_index in self.mesh.get_forcing_pointer_cells():
            for (face_index, orientation) in \
                    self.mesh.get_forcing_pointers_for_cell(cell_index):
                coupling_data.append(-self.mesh.get_face_area(face_index) *
                                      orientation)
                coupling_row.append(cell_index +
                                    self.mesh.get_number_of_faces())
                coupling_col.append(face_index)

        for face_index in self.mesh.get_dirichlet_pointer_faces():
            (cell_index, orientation) = \
                self.mesh.get_dirichlet_pointer(face_index)
            coupling_data.append(self.mesh.get_face_area(face_index) *
                                 orientation)
            coupling_row.append(face_index)
            coupling_col.append(cell_index+self.mesh.get_number_of_faces())

        for face_index in self.mesh.get_all_face_to_lagrange_pointers():
            (lagrange_index, orientation) = \
                self.mesh.get_face_to_lagrange_pointer(face_index)
            coupling_data.append(self.mesh.get_face_area(face_index) *
                                 orientation)
            coupling_row.append(face_index)
            coupling_col.append(lagrange_index)

        for lagrange_index in self.mesh.get_all_lagrange_to_face_pointers():
            for (face_index, orientation) in \
                    self.mesh.get_lagrange_to_face_pointers(lagrange_index):
                coupling_data.append(-self.mesh.get_face_area(face_index) *
                                      orientation)
                coupling_row.append(lagrange_index)
                coupling_col.append(face_index)

        return [coupling_data, coupling_row, coupling_col]

    def build_lhs(self, alpha = 0):
        """ Builds the entire saddle point problem,
        |  M    DIV^T |
        |             |
        | DIV     C   |
        Returns the system in coo matrix format.
        """
        lhs_data = array.array('f')
        lhs_row = array.array('i')
        lhs_col = array.array('i')

        m_info = self.build_m()

        lhs_data += m_info[0]
        lhs_row += m_info[1]
        lhs_col += m_info[2]

        del m_info

        [div_info, div_t_info] = self.build_div()

        lhs_data += div_info[0]
        lhs_row += div_info[1]
        lhs_col += div_info[2]

        del div_info

        lhs_data += div_t_info[0]
        lhs_row += div_t_info[1]
        lhs_col += div_t_info[2]

        del div_t_info

        c_info = self.build_bottom_right(alpha)

        lhs_data += c_info[0]
        lhs_row += c_info[1]
        lhs_col += c_info[2]

        del c_info

        coupling_info = self.build_coupling_terms()

        lhs_data += coupling_info[0]
        lhs_row += coupling_info[1]
        lhs_col += coupling_info[2]

        del coupling_info

        self.lhs = sparse.coo_matrix((lhs_data,
                                      (lhs_row,
                                       lhs_col)))

        del lhs_data
        del lhs_row
        del lhs_col

        return self.lhs

    def build_lhs_petsc(self, alpha = 0):
        """
        Builds the entire saddle point problem,
        |  M    DIV^T |
        |             |
        | DIV     C   |
        Returns the system in coo matrix format.
        """
        self.build_lhs()
        lhs_csr = self.lhs.tocsr()
        self.lhs = PETSc.Mat()

        self.lhs.create(PETSc.COMM_WORLD)
        self.lhs.createAIJWithArrays(size=lhs_csr.shape, csr=(lhs_csr.indptr,
                                                               lhs_csr.indices,
                                                               lhs_csr.data))

        self.lhs.setUp()

        self.lhs.assemblyBegin()
        self.lhs.assemblyEnd()

        return self.lhs

        m_info = self.build_m()

        self.m = sparse.coo_matrix((m_info[0], (m_info[1], m_info[2]))).tocsr()

        total_work = len(m_info[0])
        percentage_inc = 10.
        last_percent = 10.

        for index in range(len(m_info[0])):
            if float(index)/float(total_work)*100>last_percent:
                print "percentage done", last_percent
                last_percent+= percentage_inc

            self.lhs[m_info[1][index], m_info[2][index]] = m_info[0][index]

        del m_info

        [div_info, div_t_info] = self.build_div()

        for index in range(len(div_info[0])):
            self.lhs[div_info[1][index], div_info[2][index]] = \
                div_info[0][index]

        for index in range(len(div_t_info[0])):
            self.lhs[div_t_info[1][index], div_t_info[2][index]] = \
                div_t_info[0][index]

        del div_info
        del div_t_info

        c_info = self.build_bottom_right(alpha)

        for index in range(len(c_info[0])):
            self.lhs[c_info[1][index], c_info[2][index]] = c_info[0][index]

        del c_info

        coupling_info = self.build_coupling_terms()

        for index in range(len(coupling_info[0])):
            self.lhs[coupling_info[1][index], coupling_info[2][index]] = \
                coupling_info[0][index]

        del coupling_info


    def build_lhs_divided(self, alpha = 0):
        """ Builds the saddle point problem,
        |  M    DIV^T |
        |             |
        | DIV     C   |
        divided into the separate components.
        """
        m_info = self.build_m()

        self.m = sparse.coo_matrix((m_info[0], (m_info[1], m_info[2]))).tocsr()

        del m_info

        [div_info, div_t_info] = self.build_div(shift=0)

        self.div = sparse.coo_matrix((div_info[0],
                                      (div_info[1],
                                       div_info[2])))

        self.div_t = sparse.coo_matrix((div_t_info[0],
                                        (div_t_info[1],
                                         div_t_info[2])))

        del div_info
        del div_t_info

        c_info = self.build_bottom_right(alpha)

        self.c = sparse.coo_matrix((c_info[0],
                                    (c_info[1], c_info[2])))

        del c_info

        coupling_info = self.build_coupling_terms()

        if coupling_info[0]:
            self.coupling = sparse.coo_matrix((coupling_info[0],
                                               (coupling_info[1],
                                                coupling_info[2])))
        else:
            pass

        del coupling_info

    def get_number_of_dof(self):
        """ Returns total number of degrees
        of freedom for the saddle-point system.
        """
        return (self.mesh.get_number_of_cells()+
                self.mesh.get_number_of_faces())

    def build_rhs(self):
        """ Build RHS for the saddle-point problem.
        """
        self.rhs = np.zeros(self.get_number_of_dof())

        self.build_rhs_neumann()
        self.build_rhs_dirichlet()
        self.build_rhs_forcing()
        return self.rhs

    def add_gravity(self):
        """ Adds gravity effects to right hand side of problem
        based on density, acceleration and gravity vector
        set in Mesh class.
        """
        gravity_force = np.zeros(self.mesh.get_number_of_faces())
        for face_index in range(self.mesh.get_number_of_faces()):
            gravity_force[face_index] = self.mesh.get_fluid_density()
            gravity_force[face_index] *= self.mesh.get_gravity_acceleration()
            gravity_force[face_index] *= \
                np.dot(self.mesh.get_gravity_vector(),
                       self.mesh.get_face_normal(face_index))

        [m_data, m_row, m_col] = self.build_m(k_unity = True)
        m_coo = sparse.coo_matrix((np.array(m_data),
                                   (np.array(m_row),
                                    np.array(m_col))))

        gravity_force = m_coo.dot(gravity_force)
        neumann_boundary_indices = \
            map(lambda x: x[0], self.get_neumann_boundary_values())

        for face_index in range(self.mesh.get_number_of_faces()):
            if face_index not in neumann_boundary_indices:
                self.rhs[face_index] += gravity_force[face_index]

    def build_rhs_forcing(self):
        """ Construct the entries for the forcing
        function in the RHS.
        """
        for cell_index in range(self.mesh.get_number_of_cells()):
            self.rhs[cell_index + self.mesh.get_number_of_faces()] +=\
                self.cell_forcing_function[cell_index]

    def build_rhs_neumann(self):
        """ Construct the entries for the Neumann boundaries
        in the RHS.
        """
        for [boundary_index, boundary_value] in \
                self.get_neumann_boundary_values():
            self.rhs[boundary_index] =  boundary_value

    def get_cell_faces_neumann(self, cell_index):
        """ Returns all the faces in cell_index that
        are also neumann faces.
        """
        if self.cell_faces_neumann.has_key(cell_index):
            return self.cell_faces_neumann[cell_index]
        else:
            return []

    def apply_neumann_from_function(self, boundary_marker, grad_u):
        """ Sets the Neumann boundary values for the
        faces assigned to boundary_marker based on a
        a functional representation grad_u. All
        boundary faces associated with
        boundary_marker will be set as Neumann
        The function grad_u takes a Numpy array
        representing a point coordinate on the
        face and returns
        the vector of the Neumann condition at that point.
        The face centroid is used as a quadrature point.
        """
        for [boundary_index, boundary_orientation] in \
                self.mesh.get_boundary_faces_by_marker(boundary_marker):
            current_normal = self.mesh.get_face_normal(boundary_index)
            current_centroid = self.mesh.get_face_real_centroid(boundary_index)
            current_grad = grad_u(current_centroid)
            neumann_value = current_grad.dot(current_normal)

            self.neumann_boundary_values[boundary_index] = neumann_value
            cell_index = self.mesh.face_to_cell[boundary_index][0]

            cell_list = list(self.mesh.get_cell(cell_index))
            local_face_index = cell_list.index(boundary_index)

            if self.cell_faces_neumann.has_key(cell_index):
                self.cell_faces_neumann[cell_index].append(boundary_index)
            else:
                self.cell_faces_neumann[cell_index] = [boundary_index]


    def set_neumann_by_face(self,
                              face_index,
                              face_orientation,
                              value):
        """ Directly sets Neumann value to face_index.
        The input *value* corresponding to the
        integral of the pressure over the face.
        """
        self.neumann_boundary_values[face_index] = value

        cell_index = self.mesh.face_to_cell[face_index][0]

        if self.cell_faces_neumann.has_key(cell_index):
            self.cell_faces_neumann[cell_index].append(face_index)
        else:
            self.cell_faces_neumann[cell_index] = [face_index]

    def get_dirichlet_boundary_values(self):
        """ Returns a list of all faces designated as
        Dirichlet boundaries and their values.
        """
        return self.dirichlet_boundary_values.iteritems()

    def get_dirichlet_boundary_value_by_face(self, face_index):
        """ Return pressure value at face_index.
        """
        return self.dirichlet_boundary_values[face_index]

    def get_number_of_dirichlet_faces(self):
        """ Returns the number of Dirichlet boundary faces.
        """
        return len(self.dirichlet_boundary_values)

    def get_neumann_boundary_values(self):
        """ Returns a list of all faces designated as
        Neumann boundaries and their values.
        """
        return self.neumann_boundary_values.iteritems()

    def reset_dirichlet_boundaries(self):
        """ Resets all Dirichlet bounday data.
        """
        self.dirichlet_boundary_values = {}

    def reset_neumann_boundaries(self):
        """ Resets all Neumann bounday data.
        """
        self.neumann_boundary_values = {}

    def apply_dirichlet_from_function(self, boundary_marker, p_function):
        """ Sets the Dirichlet boundary values for the
        for the faces assigned to boundary_marker based on
        a functional representation p_function. All
        boundary faces associated with
        boundary_marker will be set as Dirichlet.
        The p_function takes a Numpy array
        representing a point coordinate on the
        face and returns the scalar value of
        the Dirichlet condition at that point.
        The face centoid is used as a quadrature point.
        """
        for (boundary_index, boundary_orientation) \
                in self.mesh.get_boundary_faces_by_marker(boundary_marker):
            current_centroid = self.mesh.get_face_real_centroid(boundary_index)
            dirichlet_value = p_function(current_centroid)
            dirichlet_value *= self.mesh.get_face_area(boundary_index)

            self.dirichlet_boundary_values[boundary_index] = \
                dirichlet_value*boundary_orientation

    def apply_forcing_from_function(self, forcing_function):
        """ Sets forcing function for entire problem based on
        a functional representation forcing_function.
        The forcing_function takes a Numpy array
        representing a point coordinate and returns
        the value of the forcing function at that point.
        The cell centroid is used as a quadrature point.
        """
        for cell_index in range(self.mesh.get_number_of_cells()):
            current_centroid = self.mesh.get_cell_real_centroid(cell_index)
            forcing_value = forcing_function(current_centroid)
            forcing_value *= self.mesh.get_cell_volume(cell_index)
            self.cell_forcing_function[cell_index] = forcing_value

    def apply_forcing_from_grad(self, grad_p):
        """ Computes the source term for a cell from the
        exact representation of the gradient integrated over the
        boundaries of the cell.
        """
        for cell_index in range(self.mesh.get_number_of_cells()):
            face_list = self.mesh.get_cell(cell_index)
            orientation_list = self.mesh.get_cell_normal_orientation(cell_index)
            for (face_index, orientation) in zip(face_list, orientation_list):
                current_normal = self.mesh.get_face_normal(face_index)
                current_center = self.mesh.get_face_real_centroid(face_index)
                exact_flux = np.dot(grad_p(current_center), current_normal)
                exact_flux *= self.get_face_area(face_index)
                exact_flux *= orientation
                self.cell_forcing_function[cell_index] += exact_flux

    def process_internal_boundaries(self):
        """ Conducts appropriate processing needed
        to correctly incorporate internal boundary conditions.
        """
        for (face_index, face_orientation) in self.mesh.get_internal_no_flow():
            cell_index = self.mesh.face_to_cell[face_index][0]
            cell_list = list(self.mesh.get_cell(cell_index))
            local_face_index = cell_list.index(face_index)
            if self.cell_faces_neumann.has_key(cell_index):
                self.cell_faces_neumann[cell_index].append(face_index)
            else:
                self.cell_faces_neumann[cell_index] = [face_index]
            
            self.neumann_boundary_values[face_index] = 0.
            
    def build_rhs_dirichlet(self):
        """ Construct the entries for the Dirichlet boundaries
        in the RHS.
        """
        for [boundary_index, boundary_value] in \
                self.get_dirichlet_boundary_values():
            self.rhs[boundary_index] = -boundary_value

    def get_analytical_pressure_solution(self,
                                         p_function,
                                         quadrature_method = 0):
        """ Returns the average pressure over each cell computed
        by using the real centroid of the cell as quadrature.
        """
        if quadrature_method == 0:
            p_array = [p_function(point) for
                       point in self.mesh.get_all_cell_real_centroids()]
        elif quadrature_method == 1:
            p_array = [p_function(point) for
                       point in self.mesh.get_all_cell_shifted_centroids()]

        return p_array

    def get_analytical_velocity_solution(self, grad_p):
        """ Returns the average flux over each face computed
        by using the real centroid of the face as quadrature.
        """
        flux_vector = []
        for face_index in range(self.mesh.get_number_of_faces()):
            current_normal = self.mesh.get_face_normal(face_index)
            current_center = self.mesh.get_face_real_centroid(face_index)

            flux_vector.append(np.dot(grad_p(current_center), current_normal))

        return flux_vector

    def l2_error_velocity_per_cell(self, grad_p):
        """ Returns a vector of length number of cells
        with the relative error of the velocity
        for each cell.
        """
        error_flux = np.zeros(self.mesh.get_number_of_cells())

        for cell_index in range(self.mesh.get_number_of_cells()):
            error_flux_numerator = 0.
            error_flux_denominator = 0.
            for face_index in self.mesh.get_cell(cell_index):
                current_normal = self.mesh.get_face_normal(face_index)
                current_center = self.mesh.get_face_real_centroid(face_index)

                exact_flux = np.dot(grad_p(current_center), current_normal)

                error_flux_numerator += \
                    (self.mesh.get_cell_volume(cell_index)*
                     (self.solution[face_index]-exact_flux)**2)

                error_flux_denominator += \
                    (self.mesh.get_cell_volume(cell_index)*(exact_flux)**2)

            error_flux[cell_index] = error_flux_numerator/error_flux_denominator
            error_flux[cell_index] = np.sqrt(error_flux)

        return error_flux

    def compute_l2_error_velocity(self, grad_p, quadrature_method = 0):
        """ Computes the L2 velocity error of the final
        solution relative to an analytical solution
        grad_p.
        """
        error_flux_numerator = 0.

        for cell_index in range(self.mesh.get_number_of_cells()):
            for face_index in self.mesh.get_cell(cell_index):
                current_normal = self.mesh.get_face_normal(face_index)
                if quadrature_method == 0:
                    current_center = \
                        self.mesh.get_face_real_centroid(face_index)
                elif quadrature_method == 1:
                    current_center = \
                        self.mesh.get_face_shifted_centroid(face_index)
                else:
                    raise Exception("no quadrature for method" +
                                    str(quadrature_method))

                exact_flux = np.dot(grad_p(current_center), current_normal)

                error_flux_numerator += \
                    (self.mesh.get_cell_volume(cell_index)*
                     (self.solution[face_index]-exact_flux)**2)

        return np.sqrt(error_flux_numerator)

    def compute_x_error_velocity(self, grad_p):
        """ Compute the velocity error in the X norm
        ||| grad_p^I - v_h |||_X
        """
        grad_p_i_min_v_h = np.zeros(self.mesh.get_number_of_faces() +
                                    self.mesh.get_number_of_cells())

        for face_index in range(self.mesh.get_number_of_faces()):
            current_centroid = self.mesh.get_face_real_centroid(face_index)
            current_normal = self.mesh.get_face_normal(face_index)
            grad_p_i_min_v_h[face_index] = np.dot(grad_p(current_centroid),
                                                  current_normal)

            grad_p_i_min_v_h[face_index] -= self.solution[face_index]

        error = self.lhs.dot(grad_p_i_min_v_h)[:self.mesh.get_number_of_faces()]
        error = error.dot(grad_p_i_min_v_h[:self.mesh.get_number_of_faces()])

        return np.sqrt(error)

    def compute_l2_error_pressure(self, p_function, quadrature_method = 0):
        """ Computes the L2 error against a analytical solution p_function.
        The option quadrature_method specifies the way integral is
        approximated:
        0 => use the real cell centroid.
        1 => use the shifted cell centroid.
        2 => use quadrature points specified in mesh.
        """
        error_numerator = 0.
        error_denominator = 0.

        for cell_index in range(self.mesh.get_number_of_cells()):
            if quadrature_method == 0:
                current_center = self.mesh.get_cell_real_centroid(cell_index)
                exact_pressure = p_function(current_center)

            elif quadrature_method == 1:
                current_center = self.mesh.get_cell_shifted_centroid(cell_index)
                exact_pressure = p_function(current_center)

            error_numerator += self.mesh.get_cell_volume(cell_index)* \
                (self.solution[self.mesh.get_number_of_faces()+cell_index]- \
                     exact_pressure)**2
            error_denominator += self.mesh.get_cell_volume(cell_index)* \
                exact_pressure**2

        return np.sqrt(error_numerator)

    def solve(self, initial_guess = 0., solver = 'spsolve'):
        """ Solves the full saddle points system after construction
        of the LHS and RHS of the problem. Returns the full
        solution vector.
        """
        self.solution = np.zeros(self.mesh.get_number_of_faces() +
                                 self.mesh.get_number_of_cells())
        if solver == 'spsolve':
            self.lhs = self.lhs.tocsc()
            self.solution = linalg.spsolve(self.lhs, self.rhs)

        elif solver == 'gmres':
            self.lhs = self.lhs.tocsc()
            [self.solution, converged] = linalg.gmres(self.lhs,
                                                      self.rhs,
                                                      tol=1.e-8)
            if converged > 0:
                print "did not converge after", converged

        return self.solution

    def cg_inner(self, apply_lhs, b, tol = 1.e-8):
        """ Local implementation of CG algorithm.
        Code from example in wikipedia page on CG methods.
        """
        current_x = np.zeros(len(b))
        r = b-apply_lhs(current_x)
        p = np.array(r)

        rsold = r.dot(r)
        if rsold<tol:
            return current_x

        for i in range(1000):
            print "\t\t inner iter", i
            Ap = apply_lhs(p)
            pAp = p.dot(Ap)
            if pAp<1.e-42:
                #print "return from pAp"
                return current_x

            alpha = rsold/p.dot(Ap)

            current_x += alpha*p
            r-=alpha*Ap
            rsnew = r.dot(r)
            if rsnew<tol:
                return current_x
            p=r+rsnew/rsold*p
            rsold = rsnew

        print "\t\t went over max iterations", rsnew
        return current_x

    def solve_divided(self):
        """ Solve the divided system of equations using
        a Schur compliment method.
        """
        f1 = self.rhs[:self.mesh.get_number_of_faces()]
        f2 = self.rhs[self.mesh.get_number_of_faces():]

        current_p = np.zeros(self.mesh.get_number_of_cells())


        def apply_lhs(x):
            return -self.div.dot(self.cg_inner(self.m.dot,
                                         self.div_t.dot(x),
                                         tol=1.e-16))

        f1_tilde = linalg.cg(self.m, f1)[0]

        cg_rhs = f2-self.div.dot(f1_tilde)

        current_p = self.cg(apply_lhs, cg_rhs)
        print "solver residual=>", np.linalg.norm(apply_lhs(current_p)-cg_rhs)

        current_v = -self.cg(self.m.dot,
                               self.div_t.dot(current_p),
                               tol=1.e-10)+f1_tilde

        self.solution = np.concatenate((current_v, current_p))

    def solve_petsc(self):
        """ Solving using PETSc, this requires the
        use of build_lhs_petsc.
        """
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)

        ksp.setType('gmres')
        ksp.getPC().setType('ilu')

        x, b = self.lhs.getVecs()

        x.set(0)
        b.setArray(self.rhs)

        ksp.setOperators(self.lhs)
        ksp.setFromOptions()
        print "solving..."
        ksp.solve(b, x)
        self.solution = x.getArray()

    def get_pressure_solution(self):
        """ Returns the pressure solution for the problem. The
        pressures are indexed in the same order as the
        corresponding cell indices.
        """
        return self.solution[self.mesh.get_number_of_faces():]

    def get_velocity_solution_by_index(self, face_index):
        """
        Returns the flux for face index by face_index.
        """
        return self.solution[face_index]

    def get_velocity_solution(self):
        """
        Returns the velocity solution for the problem. The
        fluxes are indexed in the same order as the
        corresponding face indices.
        """
        return self.solution[:self.mesh.get_number_of_faces()]

    def lhs_rank(self):
        """
        Return the rank of the saddle-point problem
        using the SVD.
        """
        lhs = self.lhs.todense()
        return len(filter(lambda x:abs(x)>1.e-9,np.linalg.svd(lhs)[1]))

    def print_lhs_to_file(self, filename):
        """
        Output entire LHS to file (dense format).
        """
        matout = open( filename + ".dat", 'w')

        lhs = self.lhs.tocsr()

        (height, width) = lhs.get_shape()
        for i in range(height):
            for j in range(width):
                print >> matout, lhs[i, j], ",",
            print >> matout

        matout.close()

    def print_lhs_to_ppm(self, filename):
        """
        Output LHS to ppm image file type.
        """
        matout = open( filename + ".ppm", 'w')

        (height, width) = self.lhs.get_shape()

        print >> matout, "P1"
        print >> matout, height, width

        for i in range(height):
            for j in range(width):
                if abs(self.lhs[i,j]) > 1.e-10:
                    print >> matout, 1,
                else:
                    print >> matout, 0,
            print >> matout

        matout.close()
