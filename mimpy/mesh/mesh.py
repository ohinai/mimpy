""" Mesh module.
"""
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import array
from itertools import islice
import mimpy.mesh.mesh_cython as mesh_cython
import mimpy as mimpy
from six.moves import map
from six.moves import range
from six.moves import zip

def tb(s):
    return bytes(s, "UTF-8")

class variable_array():
    """ The class is an efficient reprenstation of variable
    lenght two dimensional arrays. It can represent
    basic data types such as ints and floats and allows variable
    lengths on entries. That is:

    a[0] = [1, 2, 3, 4,]
    a[1] = [1, 2, 3, 4, 5, 6, 7]

    Internally, the data is stored is a 1-d array, with a
    separate array indicating the offset and data lenth:

    data = [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7]
    offset = [[0, 4], [5, 7]]

    The structure allows the user to modify the entries
    as well as extend the data as needed.
    """
    def __init__(self, dtype=np.dtype('d'), size=(2, 2), dim = 1):
        self.pointer_capacity = size[0]
        self.data_capacity  = size[1]
        self.dim = dim

        self.dtype = dtype
        self.pointers = np.empty(shape=(self.pointer_capacity, 2),
                                 dtype=np.dtype("i"))
        if self.dim == 1:
            self.data = np.empty(shape = (self.data_capacity),
                                 dtype=self.dtype)
        else:
            self.data = np.empty(shape = (self.data_capacity, self.dim),
                                 dtype=self.dtype)

        self.number_of_entries = 0
        self.next_data_pos = 0

    def __del__(self):
        del self.pointers
        del self.data

    def set_pointer_capacity(self, capacity):
        """ Sets the maximum number of entries in
        the data structure.

        :param int capacity: Number of expected entries.

        :return: None
        """
        self.pointer_capacity = capacity
        self.pointers.resize((self.pointer_capacity, 2), refcheck = False)

    def set_data_capacity(self, capacity):
        """ Sets the maximum number of entries in
        the data structure.

        :param int capacity: Total data entries.

        :return: None
        """
        self.data_capacity = capacity
        if self.dim == 1:
            self.data.resize((self.data_capacity), refcheck=False)
        else:
            self.data.resize((self.data_capacity, self.dim), refcheck=False)

    def new_size(self, size, minimum = 1000):
        """ Calculates the new size of the array
        given the old in case there is a need
        for extending the array.

        :param int size: Old data size.
        :param int minimum: Sets minimum new data size.

        :return: New data structure size.
        :rtype: int
        """
        return max(size+size/2+2, minimum)

    def add_entry(self, data):
        """ Adds new data to end of the list.

        :param  dtype data: Generic data to be added. Usually
        either scalar type (float, int ...) or ndarray type.

        :return: Index of the new entry.
        :rtype: int
        """
        if self.number_of_entries < len(self.pointers):
            self.pointers[self.number_of_entries, 0] = self.next_data_pos
            self.pointers[self.number_of_entries, 1] = len(data)
        else:
            new_array_size = self.new_size(len(self.pointers), len(data))
            self.pointers.resize((new_array_size, 2),
                                 refcheck=False)
            self.pointers[self.number_of_entries, 0] = self.next_data_pos
            self.pointers[self.number_of_entries, 1] = len(data)

        if len(self.data) > self.next_data_pos+len(data):
            self.data[self.next_data_pos:self.next_data_pos+len(data)] = data
        else:
            if self.dim == 1:
                self.data.resize(self.new_size(len(self.data)),
                                 refcheck=False)
            else:
                self.data.resize((self.new_size(len(self.data)), self.dim),
                                 refcheck=False)
            self.data[self.next_data_pos:self.next_data_pos+len(data)] = data

        self.next_data_pos = len(data)+self.next_data_pos
        self.number_of_entries += 1

        return self.number_of_entries-1

    def get_entry(self, index):
        """ Return entry.
        """
        if index > self.number_of_entries:
            raise IndexError("No entry with index " +str(index))
        (pos,  length) = self.pointers[index]
        return self.data[pos:pos+length]

    def __getitem__(self, index):
        """Overloaded get index function.
        """
        return self.get_entry(index)

    def __setitem__(self, index, value):
        """ Overloaded setting function.
        """
        self.set_entry(index, value)

    def __len__(self):
        """ Returns number of entries.
        """
        return self.number_of_entries

    def set_entry(self, index, data):
        """ Changes existing entry to new data.
        The new entry can be larger than old, but might cause
        wasted memory.
        """
        (pos, length) = self.pointers[index]
        if length >= len(data):
            self.data[pos:pos+len(data)] = data
            self.pointers[index, 1] = len(data)
        else:
            if len(self.data) > self.next_data_pos+len(data):
                self.data[self.next_data_pos:
                              self.next_data_pos+len(data)] = data
            else:
                if self.dim == 1:
                    self.data.resize(self.new_size(len(self.data)),
                                     refcheck=False)
                    self.data[self.next_data_pos:
                                  self.next_data_pos+len(data)] = data
                else:
                    self.data.resize(self.new_size((len(self.data)), self.dim),
                                     refcheck=False)
                    self.data[self.next_data_pos:
                                  self.next_data_pos+len(data)] = data

            self.pointers[index, 0] = self.next_data_pos
            self.pointers[index, 1] = len(data)
            self.next_data_pos += len(data)

class Mesh:
    """ The **Mesh** class is a common representation of polygonal
    meshes in Mimpy. In addition to the mesh data structure,
    it provides commonly used mesh functions as such
    calculating volumes and centroids as well as basic visualization. 
    The **Mesh** class serves as base implementation, 
    with the specific mesh types (such as hexahedra,
    tetrahedra and Voronoi) inherting from it.
    """
    def __init__(self):
        # List of points used to construct mesh faces.
        # Each point coordinate is prepresented by
        # a Numpy array.
        self.points = np.empty(shape=(0, 3), dtype=np.dtype('d'))
        self.number_of_points = 0

        # List of mesh faces, each face is represented by the
        # a list of points. In 2D, it's a list of pairs of poitns.
        # In 3D, it's an ordered list of points that make up the
        # polygon.
        self.faces = variable_array(dtype=np.dtype('i'))

        # Face normals.
        self.face_normals = np.empty(shape=(0, 3), dtype=np.dtype('d'))

        # Area of mesh face.
        self.face_areas = np.empty(shape=(0), dtype=np.dtype('d'))

        # The centroid of face.
        self.face_real_centroids = np.empty(shape=(0, 3))

        # Dict that maps faces to the cells
        # they are in.
        self.face_to_cell = np.empty(shape=(0, 2), dtype=np.dtype('i'))

        # A point on the plane of the face that is used
        # to build the MFD matrix R. This point does
        # not have to be on the face itself.
        self.face_shifted_centroids = np.empty(shape=(0, 3))

        self.has_face_shifted_centroid = False
        self.has_cell_shifted_centroid = False

        self.has_alpha = False

        self.boundary_markers = []
        self.boundary_descriptions = []

        # Hash from face marker => [[face index, face normal orientation]]
        self.boundary_faces = {}

        # List of cells. Each cell is made up of a list
        # of faces.
        self.cells = variable_array(dtype=np.dtype('i'))

        # For each cell, a list of bools indicating
        # whether the normal in self.face_normals
        # is in or out of the cell.
        self.cell_normal_orientation  = variable_array(dtype=np.dtype('i'))

        # Cell Volumes.
        self.cell_volume = np.empty(shape=(0), dtype=np.dtype('d'))

        # List of cell centroids.
        self.cell_real_centroid = np.empty(shape=(0, 3), dtype=np.dtype('d'))

        # Points used inside the cell used
        # to build the MFD matrix R.
        self.cell_shifted_centroid = np.empty(shape=(0, 3), dtype=np.dtype('d'))

        self.cell_k = np.empty(shape=(0, 9))

        # Tags cells depending on which domain
        # they belong to (for fractures and
        # multi-domain problems)
        self.cell_domain = np.empty(shape=(0), dtype=int)

        self.dim = 3

        # dict: {face_index: (cell_index, face_orientation), ...}
        # Allows Dirichlet boundaries to be set implicitly
        # based on pressure of cells.
        self.dirichlet_boundary_pointers = {}

        # Faces designated as no flow, meant for
        # interior boundary conditions not to be
        # set by user.
        self.internal_no_flow = []

        # dict: {face_index: (lagrange_index, orientation)}
        # Allows dirichlet boundaries to point to
        # lagrange multipliers for domain decomposition.
        # A lagrange multiplier is a face identicial
        # to the one pointing to it, but not associated
        # with any cells.
        self.face_to_lagrange_pointers = {}

        # dict: lagrange_index: [(face_index_1, orientation), ...], ...}
        # Points a lagrange multiplier to faces associated
        # with it. Is treated like a forcing function.
        self.lagrange_to_face_pointers = {}

        # dict: {cell_index: [(face_index_1, orientation), ...], ...}
        # Allows source terms to be set implicitly
        # based on fluxes at other faces.
        self.forcing_function_pointers = {}

        # Lowest order term coef
        # List: [alpha1, alpha2, ...]
        self.cell_alpha = []
        self.is_using_alpha_list = False

        self.gravity_vector = None
        self.gravity_acceleration = 9.8

    def add_point(self, new_point):
        """ Takes a Numpy array
        representing the cartesian point coodrinates,
        and appends the point to the end of the point list.
        Returns the index of the new point.

        :param ndarray new_point: New point to be added to mesh.
        :return: Index of new point.
        :rtype: int
        """
        if self.number_of_points < len(self.points):
            self.points[self.number_of_points] = new_point
            self.number_of_points += 1
        else:
            new_array_size = (len(self.points)+len(self.points)/2+1, 3)
            self.points.resize(new_array_size, refcheck=False)
            self.points[self.number_of_points] = new_point
            self.number_of_points += 1
        return self.number_of_points-1

    def get_point(self, point_index):
        """ Takes a point index and returns
        a Numpy array of point coodrinates.

        :param int point_index:

        :return: The the point coordinates.
        :rtype: ndarray
        """
        return self.points[point_index]

    def get_number_of_points(self):
        """ Returns the total number of points.

        :return: Total number of points in mesh.
        :rtype: int
        """
        return self.number_of_points

    def _memory_extension(self, size):
        """ Function for finding size of memory
        extension jumps.
        """
        return size+size/2+1

    def add_face(self, list_of_points):
        """ Takes a list of point indices, and
        appends them to the list of faces
        in the mesh. The point indices must
        be oriented in a clockwise direction
        relative to the face normal. In 2D, a
        face is represnted by two points.
        Returns the index of the new face.

        :param list list_of_points: List of point indices
             making up the new face.

        :return: Index of new face.
        :rtype: int
        """
        new_face_index = self.faces.add_entry(list_of_points)

        if len(self.face_normals)-1 < new_face_index:
            new_size = self._memory_extension(len(self.face_normals))
            self.face_normals.resize((new_size, 3), refcheck=False)

        if len(self.face_areas)-1 < new_face_index:
            new_size = self._memory_extension(len(self.face_areas))
            self.face_areas.resize(new_size, refcheck=False)

        if len(self.face_real_centroids)-1 < new_face_index:
            new_size = self._memory_extension(len(self.face_real_centroids))
            self.face_real_centroids.resize((new_size, 3), refcheck=False)

        if len(self.face_to_cell)-1 < new_face_index:
            new_size = self._memory_extension(len(self.face_to_cell))
            self.face_to_cell.resize((new_size, 2))
            self.face_to_cell[new_face_index:, :] = -1

        if self.has_face_shifted_centroid:
            if len(self.face_shifted_centroids)-1 < new_face_index:
                new_size = self._memory_extension(
                    len(self.face_shifted_centroids))
                self.face_shifted_centroids.resize((new_size, 3))

        return new_face_index

    def set_face(self, face_index, points):
        """ Sets a new set of points for a given face_index.

        :param int face_index: Face index of face to be set.
        :param list points: New list of points making up face.

        :return: None
        """
        self.faces[face_index] = points

    def remove_from_face_to_cell(self, face_index, cell_index):
        """ Removes the cell_index from face_to_cell map
        at for face_index.

        :param int face_index: Face index.
        :param int cell_index: Cell index.

        :return: None
        """
        if self.face_to_cell[face_index, 0] == cell_index:
            self.face_to_cell[face_index, 0] = -1
        elif self.face_to_cell[face_index, 1] == cell_index:
            self.face_to_cell[face_index, 1] = -1
        else:
            raise Exception("cell_index " + str(cell_index)+
                            " not found in face_to_cell for "+
                            str(face_index))

    def add_to_face_to_cell(self, face_index, cell_index):
        """ Adds cell_index to face_to_cell map
        at face_index.
        """
        if self.face_to_cell[face_index, 0] ==  -1:
            self.face_to_cell[face_index, 0] =  cell_index
        elif self.face_to_cell[face_index, 1] ==  -1:
            self.face_to_cell[face_index, 1] =  cell_index
        else:
            raise Exception("cell_index " + str(cell_index)+
                            " could not be added to "+
                            str(face_index))

    def duplicate_face(self, face_index):
        """ Creates new face with all the properties
        of the face_index, and adds the face to the
        bottom of the face list. The function
        returns the new face index.

        :param int face_index: Face index to be duplicated.
        :return: Face index of new duplicated face.
        :rtype: int
        """
        # Proper duplication requires duplicating
        # all the properties fo the face.
        new_index = self.add_face(self.get_face(face_index))
        self.set_face_area(new_index, self.get_face_area(face_index))
        return new_index

    def get_face(self, face_index):
        """ Given a face index, returns the
        list of point indices that make
        up the face.

        :param int face_index: Face index.
        :return: List of points making up face.
        :rtype: ndarray('i')
        """
        return self.faces[face_index]

    def get_number_of_face_points(self, face_index):
        """ Returns the number of points that make
        up a given face.

        :param int face_index: Face index.
        :return: Number of point making up the face.
        :rtype: int
        """
        return len(self.faces[face_index])

    def get_number_of_faces(self):
        """ Returns the total number of faces
        in the mesh. This corresponds to the
        number of velocity degrees of freedom.

        :return: Total number of faces in the mesh.
        :rtype: int
        """
        return self.faces.number_of_entries

    def get_number_of_cell_faces(self, cell_index):
        """ Returns the number of faces for cell_index

        :param int cell_index: Cell index.
        :return: Number of faces in cell.
        :rtype: int.
        """
        return len(self.cells[cell_index])

    def get_face_to_cell(self, face_index):
        """ Get list of cells connected with
        face_index.

        :param int face_index: Face index.
        :return: List of cell indices connected to the face.
        :rtype: list
        """
        f_to_c = list(self.face_to_cell[face_index])
        f_to_c = [x for x in f_to_c if x >=0]
        return f_to_c

    def is_line_seg_intersect_face(self, face_index, p1, p2):
        """ Returns True if the line segment
        intersects with a face.

        :param int face_index: Face index.
        :param ndarray p1: Coorindates of first point.
        :param ndarray p2: Coorindates of second point.
        :return: True if line segments intersects face.
        :rtype: bool
        """
        vector = p2 - p1
        vector /= np.linalg.norm(vector)

        d = np.dot((self.get_face_real_centroid(face_index) - p1),
                   self.get_face_normal(face_index))
        denom = np.dot(vector, self.get_face_normal(face_index))

        if abs(denom) < 1e-10:
            pass
        else:
            d /= denom
            length = np.linalg.norm(p1-p2)

            if d <= length+1.e-8 and d > 0.+1.e-8:
                intersection_point = d*vector+p1

                direction = np.zeros(len(self.get_face(face_index)))

                normal = self.get_face_normal(face_index)
                current_point = self.get_point(self.get_face(face_index)[-1])
                for (local_index, next_point_index) in \
                        enumerate(self.get_face(face_index)):
                    next_point = self.get_point(next_point_index)
                    face_vec = next_point - current_point
                    check_vec = current_point - intersection_point

                    direction[local_index] = np.dot(np.cross(face_vec,
                                                             check_vec),
                                                    normal)
                    current_point = next_point

                if (direction>0.).all():
                    return True
                elif (direction<0.).all():
                    return True
                else:
                    return False


    def initialize_cells(self, number_of_cells):
        """ Initialize cell data structure
        for known number of cells.
        """
        raise NotImplementedError

    def load_mesh(self, input_file):
        """ Loads mesh from mms file.

        :param file intput_file: Mesh file (mms).
        :return: None
        """
        version = next(input_file)
        date = next(input_file)
        name = next(input_file)
        comments = next(input_file)
        next(input_file)
        next(input_file)

        for line in input_file:
            line_split = line.split()
            if line_split[0] == "POINTS":
                number_of_points = int(line_split[1])
                self.number_of_points = number_of_points
                self.points = np.loadtxt(islice(input_file, number_of_points))

            elif line_split[0] == "FACES":
                number_of_faces = int(line_split[1])
                self.faces.number_of_entries = number_of_faces
                current_line = next(input_file)
                n_data_entries = int(current_line)
                self.faces.data = np.loadtxt(islice(input_file, n_data_entries),
                                             dtype=np.dtype('i'))
                current_line = next(input_file)
                n_pointers = int(current_line)
                self.faces.pointers = np.loadtxt(islice(input_file, n_pointers),
                                                 dtype=np.dtype('i'))

            elif line_split[0] == "FACE_NORMALS":
                number_of_faces = int(line_split[1])
                self.face_normals = np.loadtxt(islice(input_file,
                                                      number_of_faces))

            elif line_split[0] == "FACE_AREAS":
                number_of_faces = int(line_split[1])
                self.face_areas = np.loadtxt(islice(input_file,
                                                    number_of_faces))

            elif line_split[0] == "FACE_REAL_CENTROIDS":
                number_of_faces = int(line_split[1])
                self.face_real_centroids = np.loadtxt(islice(input_file,
                                                             number_of_faces))

            elif line_split[0] == "FACE_SHIFTED_CENTROIDS":
                self.has_face_shifted_centroid = True
                number_of_faces = int(line_split[1])
                self.face_shifted_centroids = np.loadtxt(
                    islice(input_file, number_of_faces))

            elif line_split[0] == "CELLS":
                number_of_cells = int(line_split[1])
                self.cells.number_of_entries = number_of_cells
                current_line = next(input_file)
                n_data_entries = int(current_line)
                self.cells.data = np.loadtxt(islice(input_file, n_data_entries),
                                             dtype=np.dtype('i'))

                current_line = next(input_file)
                n_pointers = int(current_line)

                self.cells.pointers = np.loadtxt(islice(input_file, n_pointers),
                                                 dtype=np.dtype('i'))

            elif line_split[0] == "CELL_NORMAL_ORIENTATION":
                number_of_cells = int(line_split[1])
                self.cell_normal_orientation.number_of_entries = \
                    number_of_cells
                current_line = next(input_file)
                n_data_entries = int(current_line)
                self.cell_normal_orientation.data = \
                    np.loadtxt(islice(input_file, n_data_entries),
                               dtype=np.dtype('i'))

                current_line = next(input_file)
                n_pointers = int(current_line)
                self.cell_normal_orientation.pointers = \
                    np.loadtxt(islice(input_file, n_pointers),
                               dtype=np.dtype('i'))

            elif line_split[0] == "CELL_VOLUMES":
                number_of_cells = int(line_split[1])
                self.cell_volume = np.loadtxt(islice(input_file,
                                                     number_of_cells))

            elif line_split[0] == "CELL_REAL_CENTROIDS":
                number_of_cells = int(line_split[1])
                self.cell_real_centroid = np.loadtxt(islice(input_file,
                                                            number_of_cells))

            elif line_split[0] == "CELL_SHIFTED_CENTROIDS":
                number_of_cells = int(line_split[1])
                self.cell_shifted_centroid = np.loadtxt(
                    islice(input_file, number_of_cells))

            elif line_split[0] == "CELL_K":
                number_of_cells = int(line_split[1])
                self.cell_k = np.loadtxt(islice(input_file, number_of_cells))

            elif line_split[0] == "BOUNDARY_MARKERS":
                number_of_boundary_markers = int(line_split[1])

                for line_index in range(number_of_boundary_markers):
                    current_line = next(input_file)
                    line_split = current_line.split()
                    entries = [int(x) for x in line_split]
                    boundary_marker = entries.pop(0)
                    self.add_boundary_marker(boundary_marker, "FROMFILE")
                    while entries:
                        self.add_boundary_face(boundary_marker,
                                               entries.pop(0),
                                               entries.pop(0))

            elif line_split[0] == "DIRICHLET_BOUNDARY_POINTERS":
                number_of_pointers = int(line_split[1])
                for line_index in range(number_of_pointers):
                    current_line = next(input_file)
                    line_split = current_line.split()
                    key = int(line_split[0])
                    cell_index = int(line_split[1])
                    orientation = int(line_split[2])
                    self.set_dirichlet_face_pointer(key,
                                                    orientation,
                                                    cell_index)

            elif line_split[0] == "INTERNAL_NO_FLOW":
                number_of_faces = int(line_split[1])
                self.internal_no_flow = list(np.loadtxt(
                        islice(input_file, number_of_faces)))

            elif line_split[0] == "FORCING_FUNCTION_POINTERS":
                number_of_cells = int(line_split[1])
                for line_index in range(number_of_cells):
                    current_line = next(input_file)
                    line_split = current_line.split()
                    cell_index = int(line_split[0])
                    entries = list(map(int, line_split[1:]))
                    face_list = []
                    orientation_list = []
                    while entries:
                        face_list.append(entries.pop(0))
                        orientation_list.append(entries.pop(0))
                    self.set_forcing_pointer(cell_index,
                                             face_list,
                                             orientation_list)

            elif line_split[0] == "FACE_TO_LAGRANGE_POINTERS":
                number_of_pointers = int(line_split[1])
                for line_index in range(number_of_pointers):
                    current_line = next(input_file)
                    line_split = current_line.split()
                    face_index = int(line_split[0])
                    lagrange_index = int(line_split[1])
                    orientation = int(line_split[2])
                    self.set_face_to_lagrange_pointer(face_index,
                                                      orientation,
                                                      lagrange_index)

            elif line_split[0] == "LAGRANGE_TO_FACE_POINTERS":
                number_of_pointers = int(line_split[1])
                for line_index in range(number_of_pointers):
                    current_line = next(input_file)
                    line_split = current_line.split()
                    lagrange_index = int(line_split[0])
                    face_index = int(line_split[1])
                    orientation = int(line_split[2])
                    self.set_lagrange_to_face_pointer(lagrange_index,
                                                      face_index,
                                                      orientation)

    def save_cell(self, cell_index, output_file):
        """ Saves individual cell in mms format.

        :param int cell_index: Cell index.
        :param file output_file: File to output cell to.

        :return: None
        """
        glob_to_loc_points = {}

        temp_mesh = self.__class__()

        current_cell = []
        current_cell_orientations = []

        for (face_index, orientation) in zip(self.get_cell(cell_index),
                              self.get_cell_normal_orientation(cell_index)):
            current_face = []
            for point_index in self.get_face(face_index):
                if point_index in glob_to_loc_points:
                    current_face.append(glob_to_loc_points[point_index])
                else:
                    current_point = self.get_point(point_index)
                    local_index = temp_mesh.add_point(current_point)
                    glob_to_loc_points[point_index] = local_index
                    current_face.append(local_index)

            new_face_index = temp_mesh.add_face(current_face)
            temp_mesh.set_face_area(new_face_index,
                                    self.get_face_area(face_index))
            temp_mesh.set_face_normal(new_face_index,
                                      self.get_face_normal(face_index))
            current_centroid = self.get_face_real_centroid(face_index)
            temp_mesh.set_face_real_centroid(new_face_index, current_centroid)
            current_cell.append(new_face_index)
            current_cell_orientations.append(orientation)

        temp_mesh.add_cell(current_cell, current_cell_orientations)
        temp_mesh.set_cell_k(0, self.get_cell_k(cell_index))
        temp_mesh.set_cell_volume(0, self.get_cell_volume(cell_index))
        current_centroid = self.get_cell_real_centroid(cell_index)
        temp_mesh.set_cell_real_centroid(0, current_centroid)

        temp_mesh.save_mesh(output_file)

    def save_mesh(self, output_file):
        """ Saves mesh file in mms format.

        :param file output_file: File to save mesh to.
        """
        print(mimpy.__version__, file=output_file)
        print(tb("date", 'UTF-8'), file=output_file)
        print("name", file=output_file)
        print("comments", file=output_file)
        print("#", file=output_file)
        print("#", file=output_file)

        print("POINTS", end=' ', file=output_file)
        print(len(self.points), file=output_file)
        np.savetxt(output_file, self.points)

        print("FACES", self.get_number_of_faces(), file=output_file)
        print(len(self.faces.data), file=output_file)
        np.savetxt(output_file, self.faces.data,  fmt='%i')
        print(len(self.faces.pointers), file=output_file)
        np.savetxt(output_file, self.faces.pointers, fmt="%i %i")

        print("FACE_NORMALS", len(self.face_normals), file=output_file)
        np.savetxt(output_file, self.face_normals)

        print("FACE_AREAS", self.get_number_of_faces(), file=output_file)
        for face_index in range(self.get_number_of_faces()):
            print(self.get_face_area(face_index), file=output_file)

        print("FACE_REAL_CENTROIDS", self.get_number_of_faces(), file=output_file)
        for face_index in range(self.get_number_of_faces()):
            current_centroid = self.get_face_real_centroid(face_index)
            print(current_centroid[0], end=' ', file=output_file)
            print(current_centroid[1], end=' ', file=output_file)
            print(current_centroid[2], file=output_file)

        if self.has_face_shifted_centroid:
            print("FACE_SHIFTED_CENTROIDS", end=' ', file=output_file)
            print(self.get_number_of_faces(), file=output_file)
            for face_index in range(self.get_number_of_faces()):
                print(self.get_face_real_centroid(face_index), file=output_file)

        print("FACE_TO_CELL", len(self.face_to_cell), file=output_file)
        np.savetxt(output_file, self.face_to_cell, fmt="%i %i")

        print("CELLS", self.get_number_of_cells(), file=output_file)
        print(len(self.cells.data), file=output_file)
        np.savetxt(output_file, self.cells.data,  fmt='%i')
        print(len(self.cells.pointers), file=output_file)
        np.savetxt(output_file, self.cells.pointers, fmt="%i %i")

        print("CELL_NORMAL_ORIENTATION", end=' ', file=output_file)
        print(self.get_number_of_cells(), file=output_file)
        print(len(self.cell_normal_orientation.data), file=output_file)
        np.savetxt(output_file, self.cell_normal_orientation.data, fmt='%i')
        print(len(self.cell_normal_orientation.pointers), file=output_file)
        np.savetxt(output_file,
                   self.cell_normal_orientation.pointers,
                   fmt="%i %i")

        print("CELL_VOLUMES", self.get_number_of_cells(), file=output_file)
        np.savetxt(output_file, self.cell_volume)

        print("CELL_REAL_CENTROIDS", self.get_number_of_cells(), file=output_file)
        np.savetxt(output_file, self.cell_real_centroid)

        if self.has_cell_shifted_centroid:
            print("CELL_SHIFTED_CENTROIDS", end=' ', file=output_file)
            print(self.get_number_of_cells(), file=output_file)
            np.savetxt(ouptut_file, self.cell_shifted_centroid)

        print("CELL_K", self.get_number_of_cells(), file=output_file)
        np.savetxt(output_file, self.cell_k)

        print("BOUNDARY_MARKERS", len(self.boundary_markers), file=output_file)
        for marker_index in self.boundary_markers:
            print(marker_index, end=' ', file=output_file)
            for (face_index, face_orientation) in\
                    self.get_boundary_faces_by_marker(marker_index):
                print(face_index, face_orientation, end=' ', file=output_file)
            print("\n", end=' ', file=output_file)

        print("DIRICHLET_BOUNDARY_POINTERS", end=' ', file=output_file)
        print(len(list(self.dirichlet_boundary_pointers.keys())), file=output_file)
        for key in self.dirichlet_boundary_pointers:
            cell_index, orientation = self.dirichlet_boundary_pointers[key]
            print(key, cell_index, orientation, file=output_file)

        print("INTERNAL_NO_FLOW", end=' ', file=output_file)
        print(len(self.internal_no_flow), file=output_file)
        np.savetxt(output_file, self.internal_no_flow)

        print("FORCING_FUNCTION_POINTERS", end=' ', file=output_file)
        print(len(list(self.forcing_function_pointers.keys())), file=output_file)
        for cell_index in self.forcing_function_pointers:
            print(cell_index, end=' ', file=output_file)
            for face_index, orientation in \
                    self.forcing_function_pointers[cell_index]:
                print(face_index, orientation, end=' ', file=output_file)
            print("\n", end=' ', file=output_file)

        print("FACE_TO_LAGRANGE_POINTERS", end=' ', file=output_file)
        print(len(list(self.face_to_lagrange_pointers.keys())), file=output_file)
        for key in self.face_to_lagrange_pointers:
            lagrange_index, orientation = self.face_to_lagrange_pointers[key]
            print(key, lagrange_index, orientation, file=output_file)

        print("LAGRANGE_TO_FACE_POINTERS", end=' ', file=output_file)
        print(len(list(self.lagrange_to_face_pointers.keys())), file=output_file)
        for key in self.lagrange_to_face_pointers:
            face_index, orientation = self.lagrange_to_face_pointers[key]
            print(key, face_index, orientation, file=output_file)

        output_file.close()

    def set_cell_faces(self, cell_index, faces):
        """ Sets the cell faces.

        :param int cell_index: Cell index.
        :param list faces: Faces making up cell.

        :return: None
        """
        self.cells[cell_index] = faces
        for face_index in faces:
            if cell_index not in self.face_to_cell[face_index]:
                self.add_to_face_to_cell(face_index, cell_index)

    def set_cell_orientation(self, cell_index, orientation):
        """ Sets the cell orientation of faces.

        :param int cell_index: Cell index.
        :paramt list orientation: List of new cell face orientations.

        :return: None
        """
        self.cell_normal_orientation[cell_index] = orientation

    def add_cell(self,
                 list_of_faces,
                 list_of_orientations):
        """ Adds a new cell to the mesh. A cell is represented
        by a list of face indices. The function also
        takes in a list of orientations of the same length
        as the list_of_faces. These represent the direction
        of the face normal relative to the cell: 1 means it
        points out, -1 means it points in to the cell.
        Returns the index of the new cell.

        :param list list_of_faces: List of face indices making up new cell.
        :param list list_of_orientations: List consisting of 1s and -1s
            indicating whether normals are pointing out (1) or in (-1) of cell.

        :return: New cell index.
        :rtype: int
        """
        new_cell_index = self.cells.add_entry(list_of_faces)
        self.cell_normal_orientation.add_entry(list_of_orientations)

        if len(self.cell_volume)-1<new_cell_index:
            new_size = self._memory_extension(len(self.cell_volume))
            self.cell_volume.resize(new_size, refcheck=False)

        if len(self.cell_k)-1<new_cell_index:
            new_size = self._memory_extension(len(self.cell_k))
            self.cell_k.resize((new_size, 9), refcheck=False)

        for face_index in list_of_faces:
            if self.face_to_cell[face_index][0] == -1:
                self.face_to_cell[face_index][0] = new_cell_index
            else:
                self.face_to_cell[face_index][1] = new_cell_index

        if len(self.cell_domain)-1<new_cell_index:
            new_size = self._memory_extension(len(self.cell_domain))
            self.cell_domain.resize(new_size, refcheck=False)

        if len(self.cell_real_centroid)-1<new_cell_index:
            new_size = self._memory_extension(len(self.cell_real_centroid))
            self.cell_real_centroid.resize((new_size, 3))

        if self.has_alpha:
            self.cell_alpha.append(None)

        len(self.cell_shifted_centroid)
        if self.has_cell_shifted_centroid:
            if len(self.cell_shifted_centroid)-1<new_cell_index:
                new_size = self._memory_extension(
                    len(self.cell_shifted_centroid))
                self.cell_shifted_centroid.resize((new_size, 3))

        return new_cell_index

    def get_cell(self, cell_index):
        """ Given a cell_index, it returns the list of faces
        that make up that cell.

        :param int cell_index: Cell index of interest.
        :return: List of faces making up cell.
        :rtype: list
        """
        return self.cells[cell_index]

    def get_cell_normal_orientation(self, cell_index):
        """ Given a cell index, returns a list of face
        orientations for that cell.

        :param int cell_index: Index of cell.
        :return: List of faces orientations in cell. The
        list is made up of 1s and -1s, 1 if the corresponding
        face normal is pointing out of the cell, and -1 if the
        corresponding face normal is pointing into the cell.

        :rtype: list
        """
        return self.cell_normal_orientation[cell_index]

    def get_number_of_cells(self):
        """ Returns total number of cells in mesh.

        :return: Number of cells in mesh.
        :rtype: int
        """
        return len(self.cells)

    def set_cell_real_centroid(self, cell_index, centroid):
        """ Sets the array of the cell centroid.

        :param int cell_index: Index of cell.
        :param ndarray centroid: New cell centroid.
        
        :return: None
        """
        self.cell_real_centroid[cell_index] = centroid

    def get_cell_real_centroid(self, cell_index):
        """ Returns array of the cell centroid
        """
        return self.cell_real_centroid[cell_index]

    def get_all_cell_real_centroids(self):
        """ Returns list of all cell centroids.
        
        :return: List of all the cell centroids.
        :rtype: ndarray
        """
        return self.cell_real_centroid[:self.get_number_of_cells()]

    def get_all_cell_shifted_centroids(self):
        """ Returns list of all cell centroids.

        :return: List of all shifted cell centroid.
        :rtype: ndarray
        """
        return self.cell_shifted_centroid[:self.get_number_of_cells()]

    def set_cell_shifted_centroid(self, cell_index, centroid):
        """ Sets the shifted centroid for cell_index.

        :param int cell_index: Index of cell.
        :param ndarray centroid: Shifted centroid point.

        :return: None
        """
        self.cell_shifted_centroid[cell_index] = centroid

    def use_face_shifted_centroid(self):
        """ Sets whether a shifted face centroid will be used
        for mesh.
        """
        self.has_face_shifted_centroid = True

    def is_using_face_shifted_centroid(self):
        """ Returns if shifted face centroids are used
        and set in mesh.

        :return: Whether face shifted centroids are set
        and used.
        :rtype: bool
        """
        return self.has_face_shifted_centroid

    def use_cell_shifted_centroid(self):
        """ Sets whether a shifted cell centroid will be used
        for mesh.
        """
        self.has_cell_shifted_centroid = True

    def is_using_cell_shifted_centroid(self):
        """ Returns if shifted face centroids are used
        and set in mesh.

        :return: Whether cell shifted centroids are set
        and used.
        :rtype: bool
        """
        return self.has_cell_shifted_centroid

    def get_cell_shifted_centroid(self, cell_index):
        """ Returns the shifted cell centroid for cell_index.

        :param int cell_index: Index of cell.
        :return: Cell shifted point.
        :rtype: ndarray
        """
        return self.cell_shifted_centroid[cell_index]

    def set_cell_volume(self, cell_index, volume):
        """ Sets cell volume for cell_index.

        :param int cell_index: Index of cell.
        :param float volume: New volume to be set for cell.

        :return: None
        """
        self.cell_volume[cell_index]  = volume

    def get_cell_volume(self, cell_index):
        """ Returns cell volume for cell_index.
        """
        return self.cell_volume[cell_index]

    def set_cell_k(self, cell_index, k):
        """ Set cell permeability tensor K
        (Numpy matrix) for cell_index.
        """
        self.cell_k[cell_index] = k.reshape((1, 9))

    def get_cell_k(self, cell_index):
        """ Return permeability tensor k
        (Numpy matrix) for cell_index.
        """
        return self.cell_k[cell_index].reshape((3, 3))

    def get_all_k_entry(self, i, j):
        """ Returns a list of all K[i, j].
        """
        return self.cell_k[:self.get_number_of_cells(), i*3+j]

    def get_all_k(self):
        """ Returns a list of all cell
        permeability tensors.
        """
        return self.cell_k

    def use_alpha(self):
        """ Activates the ability to set the
        alpha parameter for each cell.
        """
        self.has_alpha = True

    def set_alpha_by_cell(self, alpha, cell_index):
        """ Set alpha (float) for cell_index.
        """
        self.cell_alpha[cell_index] = alpha

    def get_alpha_by_cell(self, cell_index):
        """ Returns alpha (float) for cell_index.
        """
        return self.cell_alpha[cell_index]

    def set_face_real_centroid(self, face_index, centroid):
        """ Sets face centroid for face_index.
        """
        self.face_real_centroids[face_index] = centroid

    def get_face_real_centroid(self, face_index):
        """ Returns face centroid.
        """
        return self.face_real_centroids[face_index]

    def set_face_shifted_centroid(self, face_index, centroid):
        """ Sets face shifted centroid.
        """
        self.face_shifted_centroids[face_index] = centroid

    def get_face_shifted_centroid(self, face_index):
        """ Return face shifted centroid coordinates.
        """
        return self.face_shifted_centroids[face_index]

    def set_face_area(self, face_index, area):
        """ Sets face area (float) for face_index.
        """
        self.face_areas[face_index] = area

    def get_face_area(self, face_index):
        """ Return area of face.
        """
        return self.face_areas[face_index]

    def set_face_normal(self, face_index, normal):
        """ Set face normal (array) to face_index.
        """
        self.face_normals[face_index] = normal

    def get_face_normal(self, face_index):
        """ Return face normal for face_index.
        """
        return self.face_normals[face_index]

    def set_boundary_markers(self, boundary_markers, boundary_descriptions):
        """ Initialize the mesh boundary labeling. Each marker
        can represent a single boundary face or a group
        of faces.

        boundary_markers: List of integers.
        boundary_descriptions: List of strings describing
        the face groups.
        """
        self.boundary_markers = boundary_markers
        self.boundary_descriptions = boundary_descriptions

        for marker in boundary_markers:
            self.boundary_faces[marker] = []

    def add_boundary_marker(self, boundary_marker, boundary_description):
        """ Add a new boundary marker.
        """
        self.boundary_markers.append(boundary_marker)
        self.boundary_descriptions.append(boundary_description)
        self.boundary_faces[boundary_marker] = []

    def create_new_boundary_marker(self, boundary_description):
        """ Creates new boundary marker and assures
        that the index is unique. Returns the
        index of the new boundary marker.
        """
        new_index = len(self.boundary_markers)
        self.boundary_markers.append(new_index)
        self.boundary_descriptions.append(boundary_description)
        self.boundary_faces[new_index] = []
        return new_index

    def has_boundary_marker(self, boundary_marker):
        """ Returns true if boundary_marker exists.
        """
        return boundary_marker in self.boundary_markers

    def get_boundary_markers(self):
        """ Returns a list of all boundary markers.
        """
        return self.boundary_markers

    def get_boundary_discription(self, boundary_marker):
        """ Returns the boundary discription for
        boundary_marker.
        """
        return self.boundary_descriptions[boundary_marker]

    def add_boundary_face(self,
                          boundary_marker,
                          face_index,
                          face_orientation):
        """ Assign face_index to a certain boundary_marker grouping.
        the face_orientation indicates whether the normal of that
        face points in (-1) or out (1) of the cell the face
        belongs to.

        A face should never be associated with more than one marker.

        :param int boundary_marker: Boundary marker index face.
        :param int face_index: Index of face.
        :param  int face_orientation: Orientation of face normal
             relative to the domain. (1) if pointing out, (-1) if
             if pointing in.

        :return: None
        """
        self.boundary_faces[boundary_marker].append([face_index,
                                                     face_orientation])

    def set_boundary_faces(self, 
                           boundary_marker,
                           face_orientation_list):
        """ Takes a boundary_marker index, and sets the entire list
        of tuples for that boundary marker.

        :param int boundary_marker: Boundary marker to be set.
        :param list face_orienation_list: A list of tuples of the form
             [face_index, orientation] to be associated with the
             indicated boundary marker.

        :return: None
        """
        self.boundary_faces[boundary_marker] = face_orientation_list

    def get_boundary_faces_by_marker(self, boundary_marker):
        """ Returns a list of all the faces associated with a boundary_marker.

        :param int boundary_marker: Boundary marker index. 
        
        :return: List of tupes [face_index, orientation] associated with
             boundary_marker. 
        :rtype: list
        """
        return self.boundary_faces[boundary_marker]

    def is_boundary_face(self, face_index, markers):
        """ Returns True if face_index belongs to
        any of the markers.
        """
        for boundary_marker in markers:
            for face in self.boundary_faces[boundary_marker]:
                if face_index == face[0]:
                    return True

        return False

    def find_boundary_marker(self, face_index, markers):
        """ Returns the boundary marker containing
        face_index.
        """
        for boundary_marker in markers:
            for face in self.boundary_faces[boundary_marker]:
                if face_index == face[0]:
                    return boundary_marker

    def set_boundary_face_orientation(self, face_index, new_orientation):
        """ Set orientation for face_index.
        """
        for bm in self.boundary_markers:
            for face in self.boundary_faces[bm]:
                if face_index == face[0]:
                    face[1] = new_orientation

    def get_number_of_boundary_faces(self):
        """ Returns number of faces on the boundary
        of the domain.
        """
        number_of_boundary_faces = 0
        for boundary_marker in self.boundary_markers:
            number_of_boundary_faces += \
                len(self.boundary_faces[boundary_marker])

        return number_of_boundary_faces

    def add_internal_no_flow(self, face_index, face_orientation):
        """ Sets face as interior no flow boundary condition.
        """
        self.internal_no_flow.append([face_index, face_orientation])

    def get_internal_no_flow(self):
        """ Returns list of faces set as
        internal no flow condition.
        """
        return self.internal_no_flow

    def set_dirichlet_face_pointer(self,
                                   face_index,
                                   face_orientation,
                                   cell_index):
        """ Sets the value of a Dirichlet boundary to
        value of cell pressure for cell_index.
        This approach is used for coupling fractures
        with a reservoir.
        """
        # The function adds a zero entry to the
        # dirichlet_boundary_values dict. This
        # allows the MFD code to build the matrix
        # correctly, and doesn't effect the right-hand
        # side of the problem.
        self.dirichlet_boundary_pointers[face_index] = \
            (cell_index, face_orientation)

    def get_dirichlet_pointer_faces(self):
        """ Returns all the faces with Dirichlet
        values set by pointing to a cell.
        """
        return list(self.dirichlet_boundary_pointers.keys())

    def set_face_to_lagrange_pointer(self,
                                     face_index,
                                     face_orientation,
                                     lagrange_index):
        """ Sets face to dirichlet type boundary pointing to
        lagrange multiplier.
        """
        # The function adds a zero entry to the
        # dirichlet_boundary_values dict. This
        # allows the MFD code to build the matrix
        # correctly, and doesn't effect the right-hand
        # side of the problem.
        self.face_to_lagrange_pointers[face_index] = \
            (lagrange_index, face_orientation)

    def get_all_face_to_lagrange_pointers(self):
        """ Returns all face indices that are
        pointing to a lagrange multiplier.
        """
        return list(self.face_to_lagrange_pointers.keys())

    def get_face_to_lagrange_pointer(self, face_index):
        """ Returns the lagrange multiplier index
        and the face normal orientation.
        """
        return self.face_to_lagrange_pointers[face_index]

    def set_lagrange_to_face_pointers(self,
                                      lagrange_index,
                                      face_index,
                                      orientation):
        """ Sets the lagrange multiplier to the source faces
        in order to impose zero flux across the boundary.
        """
        self.lagrange_to_face_pointers[lagrange_index] = \
            [face_index, orientation]

    def get_all_lagrange_to_face_pointers(self):
        """ Returns all lagrange face indices that
        point to fluxes.
        """
        return list(self.lagrange_to_face_pointers.keys())

    def get_lagrange_to_face_pointers(self, lagrange_index):
        """ Returns the faces the lagrange_index face
        points too.
        """
        return self.lagrange_to_face_pointers[lagrange_index]

    def get_dirichlet_pointer(self, face_index):
        """ Returns the cell_index for
        which the Dirichlet boundary will be set
        implicitly.
        """
        return self.dirichlet_boundary_pointers[face_index]

    def set_forcing_pointer(self,
                            cell_index,
                            face_indices,
                            face_orientations):
        """ Sets the value of the forcing function
        implicity as the sum of the fluxes from list
        of faces. This approach is used for coupling
        fractures with a reservoir.
        """
        # The function adds a zero entry to the
        # cell_forcing_function dict. This
        # allows the MFD code to build the matrix
        # correctly, and doesn't effect the right-hand
        # side of the problem.
        # If the forcing function is set later on
        # (in case of well for example), it
        # becomes additive to the source term
        # for that cell.
        self.forcing_function_pointers[cell_index] = \
            list(zip(face_indices, face_orientations))

    def get_forcing_pointer_cells(self):
        """ Returns cell indices with forcing function
        poitners.
        """
        return list(self.forcing_function_pointers.keys())

    def get_forcing_pointers_for_cell(self, cell_index):
        """ Returns list of pointers (face_indices)
        for cell_index.
        """
        return self.forcing_function_pointers[cell_index]

    def set_cell_domain(self, cell_index, domain):
        """ Sets cell domain identifier
        for cell_index.
        """
        self.cell_domain[cell_index] = domain

    def get_cell_domain(self, cell_index):
        """ Returns cell domain identifier
        for cell_index.
        """
        return self.cell_domain[cell_index]

    def get_cell_domain_all(self):
        """ Returns list containing
        all cell_domain tags.
        """
        return self.cell_domain[:self.get_number_of_cells()]

    def get_cells_in_domain(self, domain):
        """ Returns all cells with domain tag.
        """
        cells_in_domain = []
        for cell_index in range(self.get_number_of_cells()):
            if self.cell_domain[cell_index] == domain:
                cells_in_domain.append(cell_index)
        return cells_in_domain

    def set_gravity_vector(self, gravity_vector):
        """ Set vector indicating gravity acceleration direction.
        """
        self.gravity_vector = gravity_vector

    def get_gravity_vector(self):
        """ Returns gravity vector (down direction)
        """
        return self.gravity_vector

    def get_gravity_acceleration(self):
        """ Returns the gravity acceleration constant.
        """
        return self.gravity_acceleration

    def find_basis_for_face(self, face_index):
        """ Finds two non collinear vectors
        in face to serve as basis for plane.
        """
        face = self.get_face(face_index)
        for i in range(len(face)):
            v1 = self.get_point(face[i+1]) - self.get_point(face[i])
            v2 = self.get_point(face[i]) - self.get_point(face[i-1])
            v2 /= np.linalg.norm(v2)
            v1 /= np.linalg.norm(v1)
            if 1.-abs(v1.dot(v2)) > 1.e-6:
                return (v1, v2, face[i])
        raise Exception("Couldn't compute basis for face " + str(face_index))

    def find_face_normal(self, face_index):
        """ Finds the face normal based on
        rotation around the face boundary.
        Assumes the face is planar.
        """
        face = self.get_face(face_index)
        for i in range(len(face)):
            v1 = self.get_point(face[i+1]) - self.get_point(face[i])
            v2 = self.get_point(face[i]) - self.get_point(face[i-1])
            new_face_normal = np.cross(v2, v1)
            if np.linalg.norm(new_face_normal) >1.e-10:
                new_face_normal /= np.linalg.norm(new_face_normal)
                return new_face_normal
        raise Exception("Couldn't compute normal for face " + str(face_index))

    def find_centroid_for_coordinates(self, face_index, coordinates):
        """ Computes centroid calculation for a 3D polygon based on 
        two coordinates of the polygon. 
        """
        C_1 = 0.
        C_2 = 0.
        area  = 0.
        index_1 = coordinates[0]
        index_2 = coordinates[1]

        current_face = self.get_face(face_index)
        for index in range(len(current_face)):
            current_point = self.get_point(current_face[index])
            if index == len(current_face)-1:
                next_point = self.get_point(current_face[0])
            else:
                next_point = self.get_point(current_face[index+1])

            C_1 += ((current_point[index_1]+next_point[index_1])*
                    (current_point[index_1]*next_point[index_2]-
                     next_point[index_1]*current_point[index_2]))

            C_2 += ((current_point[index_2]+next_point[index_2])*
                    (current_point[index_1]*next_point[index_2]-
                     next_point[index_1]*current_point[index_2]))

            area += current_point[index_1]*next_point[index_2]
            area -= next_point[index_1]*current_point[index_2]

        area /= 2.
        C_1 /= 6.*area
        C_2 /= 6.*area

        return (area, C_1, C_2)

    def find_face_centroid(self, face_index):
        """ Returns centroid coordinates for face_index.
        This function assumes planarity of the face.
        and is currently intended for use with three dimensional
        meshes. 
        The function returns the area of the face, as well
        as the x, y, z coordinates of its center.
        """
        (v1, v2, origin_index) = self.find_basis_for_face(face_index)
        polygon = [np.array(self.get_point(x)) for x in self.get_face(face_index)]

        assert(np.linalg.norm(v2) >1.e-12)
        assert(np.linalg.norm(v1) >1.e-12)

        v1 = v1/np.linalg.norm(v1)

        v_temp = np.cross(v1, v2)
        v2 = np.cross(v_temp, v1)

        if np.linalg.norm(v2)< 1.e-10:
            v2 = polygon[-2]-polygon[-1]
            v_temp = np.cross(v1, v2)
            v2 = np.cross(v_temp, v1)

        v2 = v2/np.linalg.norm(v2)

        origin = self.get_point(origin_index)

        transposed_polygon = [x - origin for x in polygon]
        polygon_projected_v1 = [np.dot(x, v1) for x in transposed_polygon]
        polygon_projected_v2 = [np.dot(x, v2) for x in transposed_polygon]
        polygon_projected =  list(zip(polygon_projected_v1,
                                 polygon_projected_v2))

        area = self.compute_polygon_area(polygon_projected)

        centroid_x = 0.
        centroid_y = 0.

        N = len(polygon_projected)

        for i in range(N-1):
            centroid_x += ((polygon_projected[i][0]+
                            polygon_projected[i+1][0])*
                           (polygon_projected[i][0]*
                            polygon_projected[i+1][1]-
                            polygon_projected[i+1][0]*
                            polygon_projected[i][1]))

            centroid_y += ((polygon_projected[i][1]+
                            polygon_projected[i+1][1])*
                           (polygon_projected[i][0]*
                            polygon_projected[i+1][1]-
                            polygon_projected[i+1][0]*
                            polygon_projected[i][1]))

        centroid_x += ((polygon_projected[N-1][0]+
                        polygon_projected[0][0])*
                       (polygon_projected[N-1][0]*
                        polygon_projected[0][1]-
                        polygon_projected[0][0]*
                        polygon_projected[N-1][1]))

        centroid_y += ((polygon_projected[N-1][1]+
                        polygon_projected[0][1])*
                       (polygon_projected[N-1][0]*
                        polygon_projected[0][1]-
                        polygon_projected[0][0]*
                        polygon_projected[N-1][1]))

        centroid_x = centroid_x/(6.*area)
        centroid_y = centroid_y/(6.*area)

        centroid_3d_x = 0.
        centroid_3d_y = 0.
        centroid_3d_z = 0.

        centroid_3d_x += polygon[0][0]
        centroid_3d_y += polygon[0][1]
        centroid_3d_z += polygon[0][2]

        centroid_3d_x += centroid_x * v1[0]
        centroid_3d_y += centroid_x * v1[1]
        centroid_3d_z += centroid_x * v1[2]

        centroid_3d_x += centroid_y * v2[0]
        centroid_3d_y += centroid_y * v2[1]
        centroid_3d_z += centroid_y * v2[2]

        centroid = np.array([centroid_3d_x, centroid_3d_y, centroid_3d_z])

        return (abs(area), centroid)        
                           
    def compute_polygon_area(self, polygon, dims = [0, 1]):
        """ Computes the area of a polygon. A polygon is
        represented by a list of Numpy array coordinates
        going around the polygon. The optional parameter
        *dims* represents which two coordinates to use
        when the polygon is in 3D space.
        """
        current_x = dims[0]
        current_y = dims[1]

        area = 0.

        N = len(polygon)

        for i in range(N-1):
            area += (polygon[i][current_x]*polygon[i+1][current_y]-
                     polygon[i+1][current_x]*polygon[i][current_y])

        area += (polygon[N-1][current_x]*polygon[0][current_y]-
                 polygon[0][current_x]*polygon[N-1][current_y])

        area *= .5
        # The function returns the absolute value of the area.
        return area

    def find_volume_centroid_all(self):
        """ Computes the cell centroids and volumes
        for all the cells in the mesh.
        """
        ## This is based on the code and
        ## paper by Brian Mirtich.
        zero3 = np.zeros(3)
        for cell_index in range(self.get_number_of_cells()):
            self.set_cell_volume(cell_index, 0.)
            self.set_cell_real_centroid(cell_index, zero3)

        mesh_cython.all_cell_volumes_centroids(
            self.cells.pointers,
            len(self.cells),
            self.cells.data,
            self.cell_normal_orientation.data,
            self.points,
            self.cell_volume,
            self.cell_real_centroid,
            self.faces.pointers,
            len(self.faces),
            self.faces.data,
            self.face_normals,
            self.face_to_cell)

    def find_volume_centroid(self, cell_index):
        """ Returns the volume and centroid for a 3D cell_index.
        Based on code and paper by Brian Mirtich.
        """
        volume = 0.
        centroid = np.zeros(3)
        face_list = self.get_cell(cell_index)
        orientation_list = self.get_cell_normal_orientation(cell_index)
        for (face_index, face_orientation) in zip(face_list, orientation_list):
            current_normal = self.get_face_normal(face_index)*face_orientation

            if (abs(current_normal[0]) > abs(current_normal[1])) and \
                    (abs(current_normal[0]) > abs(current_normal[2])):
                C = 0
            elif abs(current_normal[1])>abs(current_normal[2]):
                C = 1
            else:
                C = 2

            A = (C+1)%3
            B = (A+1)%3

            P1 = 0.
            Pa = 0.
            Pb = 0.
            Paa = 0.
            Pab = 0.
            Pbb = 0.

            if face_orientation > 0:
                points = self.get_face(face_index)
                next_points = list(points[1:])+list(points[:1])
            else:
                next_points = self.get_face(face_index)
                points = list(next_points[1:])+list(next_points[:1])

            for (point_index, next_point_index) in zip(points, next_points):
                a0 = self.get_point(point_index)[A]
                b0 = self.get_point(point_index)[B]
                a1 = self.get_point(next_point_index)[A]
                b1 = self.get_point(next_point_index)[B]
                da = a1-a0
                db = b1-b0
                a0_2 = a0*a0
                a0_3 = a0_2*a0
                b0_2 = b0*b0
                b0_3 = b0_2*b0
                a1_2 = a1*a1
                C1 = a1 + a0
                Ca = a1*C1 + a0_2
                Caa = a1*Ca + a0_3
                Cb = b1*(b1 + b0) + b0_2
                Cbb = b1*Cb + b0_3
                Cab = 3.*a1_2 + 2.*a1*a0 + a0_2
                Kab = a1_2 + 2*a1*a0 + 3*a0_2

                P1 += db*C1
                Pa += db*Ca
                Paa += db*Caa
                Pb += da*Cb
                Pbb += da*Cbb
                Pab += db*(b1*Cab + b0*Kab)

            P1 /= 2.0
            Pa /= 6.0
            Paa /= 12.0
            Pb /= -6.0
            Pbb /= -12.0
            Pab /= 24.0

            first_point = self.get_point(self.get_face(face_index)[0])
            w = -current_normal.dot(first_point)
            k1 = 1./current_normal[C]
            k2 = k1*k1
            k3 = k2*k1

            Fa = k1*Pa
            Fb = k1*Pb
            Fc = -k2*(current_normal[A]*Pa + current_normal[B]*Pb + w*P1)

            Faa = k1*Paa
            Fbb = k1*Pbb
            Fcc = k3*((current_normal[A]*current_normal[A])*Paa+
                      2*current_normal[A]*current_normal[B]*Pab+
                      (current_normal[B]*current_normal[B])*Pbb+
                      w*(2.*(current_normal[A]*Pa+current_normal[B]*Pb)+w*P1))

            if A == 0:
                volume += current_normal[0]*Fa
            elif B == 0:
                volume += current_normal[0]*Fb
            else:
                volume += current_normal[0]*Fc

            centroid[A] += current_normal[A]*Faa
            centroid[B] += current_normal[B]*Fbb
            centroid[C] += current_normal[C]*Fcc

        centroid /= volume*2.

        return (volume, centroid)

    def output_vector_field(self,
                            file_name,
                            vector_magnitudes = [],
                            vector_labels = []):
        """ Outputs vector data in the vtk format. The vector
        field can be processed using the glyph filter in
        Paraview. The function takes a list of vector
        magnitudes that are associated with each face
        normal.
        """
        output = open(file_name +".vtk",'wb')

        print("# vtk DataFile Version 1.0", file=output)
        print("MFD output", file=output)
        print("ASCII", file=output)
        print("DATASET UNSTRUCTURED_GRID", file=output)
        print("POINTS", self.get_number_of_faces() ,  "float", file=output)

        for face_index in range(self.get_number_of_faces()):
            current_point = self.get_face_real_centroid(face_index)
            print(current_point[0], end=' ', file=output)
            print(current_point[1], end=' ', file=output)
            print(current_point[2], file=output)

        print(" ", file=output)
        print("CELLS", self.get_number_of_faces(), end=' ', file=output)
        print(self.get_number_of_faces()*2, file=output)

        for face_index in range(self.get_number_of_faces()):
            print("1", face_index + 1, file=output)
        print(" ", file=output)
        print("CELL_TYPES" , self.get_number_of_faces(), file=output)

        for index in range(self.get_number_of_faces()):
            print("1", file=output)
        print(" ", file=output)
        print("POINT_DATA", self.get_number_of_faces(), file=output)
        print(" ", file=output)

        for data_index in range(len(vector_labels)):
            print("VECTORS", vector_labels[data_index], "float", file=output)

            for face_index in range(len(vector_magnitudes[data_index])):
                current_vector = vector_magnitudes[data_index][face_index]*\
                    self.get_face_normal(face_index)

                print(current_vector[0], end=' ', file=output)
                print(current_vector[1], end=' ', file=output)
                print(current_vector[2], file=output)

            print(" ", file=output)

    def output_cell_normals(self, file_name, cell_index):
        """ Outputs the normals over the cell in the outward direction.
        The function is intended for checking the correct orientation of cell.
        """
        output = open(file_name +".vtk",'wb')

        number_of_faces = len(self.get_cell(cell_index))

        print("# vtk DataFile Version 1.0", file=output)
        print("MFD output", file=output)
        print("ASCII", file=output)
        print("DATASET UNSTRUCTURED_GRID", file=output)
        print("POINTS", number_of_faces ,  "float", file=output)

        for face_index in self.get_cell(cell_index):
            centroid = self.get_face_real_centroid(face_index)
            print(centroid[0], end=' ', file=output)
            print(centroid[1], end=' ', file=output)
            print(centroid[2], file=output)

        print(" ", file=output)
        print("CELLS", number_of_faces, end=' ', file=output)
        print(number_of_faces*2, file=output)

        for index in range(number_of_faces):
            print("1", index+1, file=output)
        print(" ", file=output)
        print("POINT_DATA", number_of_faces, file=output)
        print(" ", file=output)

        print("VECTORS", "OUT_NORMAL", "float", file=output)
        face_list = self.get_cell(cell_index)
        orientation_list = self.get_cell_normal_orientation(cell_index)
        for (face_index, orientation) in zip(face_list, orientation_list):
            normal = self.get_face_normal(face_index)
            print(normal[0]*orientation, end=' ', file=output)
            print(normal[1]*orientation, end=' ', file=output)
            print(normal[2]*orientation, file=output)

        print(" ", file=output)

    def output_vtk_faces(self,
                         file_name,
                         face_indices,
                         face_values = [],
                         face_value_labels = []):
        """ Outputs in vtk format the faces in face_indices.
        """
        output = open(file_name +".vtk",'wb')
        print("# vtk DataFile Version 2.0", file=output)
        print("# unstructured mesh", file=output)
        print("ASCII", file=output)
        print("DATASET UNSTRUCTURED_GRID", file=output)
        print("POINTS", self.get_number_of_points(), "float", file=output)

        for point_index in range(self.get_number_of_points()):
            point = self.get_point(point_index)
            print(point[0], point[1], point[2], file=output)

        total_polygon_points = 0
        for face_index in face_indices:
            total_polygon_points += \
                self.get_number_of_face_points(face_index)+1

        print("CELLS", len(face_indices), file=output)
        print(total_polygon_points, file=output)

        for face_index in face_indices:
            current_face = self.get_face(face_index)
            print(len(current_face), end=' ', file=output)
            for point in current_face:
                print(point, end=' ', file=output)
            print("\n", end=' ', file=output)

        print("CELL_TYPES", len(face_indices), file=output)
        for face_index in face_indices:
            print(7, file=output)

        if face_values:
            print("CELL_DATA", len(face_indices), file=output)
            for (entry, entryname) in zip(face_values, face_value_labels):
                print("SCALARS", entryname, "double 1", file=output)
                print("LOOKUP_TABLE default", file=output)
                for value in entry:
                    print(value, file=output)

        output.close()

    def output_vtk_mesh(self,
                        file_name,
                        cell_values=[],
                        cell_value_labels=[]):
        """ Base implementation for producing
        vtk files for general polyhedral meshes.
        """
        output = open(file_name +".vtk",'wb')
        print("# vtk DataFile Version 2.0", file=output)
        print("# unstructured mesh", file=output)
        print("ASCII", file=output)
        print("DATASET UNSTRUCTURED_GRID", file=output)
        print("POINTS", self.get_number_of_points(), "float", file=output)

        for point_index in range(self.get_number_of_points()):
            point = self.get_point(point_index)
            print(point[0], point[1], point[2], file=output)

        total_polygon_points = 0
        for cell_index in range(self.get_number_of_cells()):
            for face_index in self.get_cell(cell_index):
                total_polygon_points += \
                    self.get_number_of_face_points(face_index)+1
            total_polygon_points += 2

        print("CELLS", self.get_number_of_cells(), end=' ', file=output)
        print(total_polygon_points, file=output)

        for cell_index in range(self.get_number_of_cells()):
            number_of_entries = len(self.get_cell(cell_index))
            for face_index in self.get_cell(cell_index):
                number_of_entries += self.get_number_of_face_points(face_index)
            number_of_entries += 1

            print(number_of_entries, end=' ', file=output)
            print(len(self.get_cell(cell_index)), end=' ', file=output)
            for face_index in self.get_cell(cell_index):
                current_face = self.get_face(face_index)
                print(len(current_face), end=' ', file=output)
                for point in current_face:
                    print(point, end=' ', file=output)
            print("\n", end=' ', file=output)

        print("CELL_TYPES", self.get_number_of_cells(), file=output)
        for cell_index in range(self.get_number_of_cells()):
            print(42, file=output)

        if cell_values:
            print("CELL_DATA", self.get_number_of_cells(), file=output)
            for (entry, entryname) in zip(cell_values, cell_value_labels):
                print("SCALARS", entryname, "double 1", file=output)
                print("LOOKUP_TABLE default", file=output)
                for value in entry:
                    print(value, file=output)

        output.close()

    def find_cell_near_point(self, point):
        """ Returns cell whose centroid is closest
        to a given point.
        """
        closest_cell = 0
        min_distance = np.linalg.norm(self.get_cell_real_centroid(0)-point)
        for cell_index in range(1, self.get_number_of_cells()):
            cell_centroid = self.get_cell_real_centroid(cell_index)
            new_distance = np.linalg.norm(cell_centroid-point)
            if new_distance < min_distance:
                closest_cell = cell_index
                min_distance = new_distance

        return closest_cell

    def subdivide_by_domain(self, cells):
        """ Takes a collection of cells, and
        seperates them from the rest of the domain
        using lagrange multipliers at the sub-domain
        boundary.
        """
        lagrange_faces = []
        for cell_index in cells:
            for face_index in self.get_cell(cell_index):
                neighboring_cells = self.face_to_cell[face_index]
                if len(neighboring_cells)> 1:
                    cell1, cell2 = neighboring_cells
                    if cell1 == cell_index:
                        if cell2 not in cells:
                            lagrange_faces.append(face_index)
                    if cell2 == cell_index:
                        if cell1 not in cells:
                            lagrange_faces.append(face_index)

        for face_index in lagrange_faces:
            (cell1, cell2) = self.face_to_cell[face_index]
            if cell1 in cells:
                other_cell = cell2
            else:
                other_cell = cell1

            new_face_index = self.add_face(self.get_face(face_index))
            lagrange_face_index = self.add_face(self.get_face(face_index))

            self.set_face_normal(new_face_index,
                                 self.get_face_normal(face_index))
            self.set_face_normal(lagrange_face_index,
                                 self.get_face_normal(face_index))

            self.set_face_real_centroid(new_face_index,
                                        self.get_face_real_centroid(face_index))
            self.set_face_real_centroid(lagrange_face_index,
                                        self.get_face_real_centroid(face_index))

            self.set_face_area(new_face_index,
                               self.get_face_area(face_index))
            self.set_face_area(lagrange_face_index,
                               self.get_face_area(face_index))

            self.add_boundary_face(100, new_face_index, 1)
            self.add_boundary_face(100, face_index, 1)

            faces_list = list(self.get_cell(other_cell))
            local_face_index_in_other = faces_list.index(face_index)

            new_cell_faces = self.get_cell(other_cell)
            new_cell_faces[local_face_index_in_other] = new_face_index

    def construct_polygon_from_segments(self, segments):
        """ Takes point pairs and constructs a single polygon
        from joining all the ends. The pairs are identified
        by directly comparing the point locations.
        """
        ## Start by setting the first point.
        current_segments = list(segments)

        new_face = [current_segments[0][0]]
        point_to_match = current_segments[0][1]

        current_segments.pop(0)

        while len(current_segments)>0:
            to_be_removed = None
            hits = 0
            for (index, segment) in enumerate(current_segments):
                if np.linalg.norm(self.get_point(point_to_match)-self.get_point(segment[0])) < 1.e-7:
                    new_face.append(segment[0])
                    to_be_removed = index
                    next_point_to_match = segment[1]
                    hits += 1
                elif np.linalg.norm(self.get_point(point_to_match)-self.get_point(segment[1])) < 1.e-7:
                    new_face.append(segment[1])
                    to_be_removed = index
                    next_point_to_match = segment[0]
                    hits += 1
            try:
                assert(hits == 1)
            except:
                for seg in current_segments:
                    print(self.get_point(seg[0]), self.get_point(seg[1]))
                raise Exception("Faild at constructing polygon from segments")

            current_segments.pop(to_be_removed)
            point_to_match = next_point_to_match

        return new_face

    def divide_cell_by_plane(self, cell_index, point_on_plane, plane_normal):
        """ Divides given cell into two cells
        based on a plane specified by a point and normal
        on the plane.
        """
        current_cell = self.get_cell(cell_index)
        interior_face_segments = []

        face_segments_to_be_added = {}
        for face_index in current_cell:
            face = self.get_face(face_index)
            new_face_1 = []
            new_face_2 = []
            face_offset = list(face[1:]) + [face[0]]
            intersection_switch = True

            for (point_index, next_point_index) in zip(face, face_offset):
                if intersection_switch:
                    new_face_1.append(point_index)
                else:
                    new_face_2.append(point_index)

                p1 = self.get_point(point_index)
                p2 = self.get_point(next_point_index)

                vector = p2 - p1
                vector /= np.linalg.norm(vector)

                d = np.dot((point_on_plane - p1), plane_normal)
                denom = np.dot(vector, plane_normal)

                if abs(denom) < 1e-10:
                    pass
                else:
                    d /= denom
                    length = np.linalg.norm(p1-p2)
                    if d <= length+1.e-8 and  d > 0.-1.e-8:
                        new_point_index = self.add_point(d*vector+p1)
                        new_face_1.append(new_point_index)
                        new_face_2.append(new_point_index)
                        if intersection_switch:
                            interior_face_segments.append([new_point_index])
                        else:
                            interior_face_segments[-1].append(new_point_index)
                        intersection_switch = not intersection_switch

            if len(new_face_2) > 0:
                self.set_face(face_index, new_face_1)
                assert(len(new_face_1)>2)
                (face_1_area, face_1_centroid)  = self.find_face_centroid(face_index)
                self.set_face_real_centroid(face_index, face_1_centroid)
                self.set_face_area(face_index, face_1_area)

                new_face_index = self.add_face(new_face_2)

                (face_area, face_centroid) = self.find_face_centroid(new_face_index)
                self.set_face_real_centroid(new_face_index, face_centroid)
                self.set_face_area(new_face_index, face_area)

                self.set_face_normal(new_face_index,
                                     self.get_face_normal(face_index))

                faces = self.get_cell(cell_index)

                self.set_cell_faces(cell_index, list(faces)+[new_face_index])

                cell_orientations = self.get_cell_normal_orientation(cell_index)
                local_face_index  = list(self.get_cell(cell_index)).index(face_index)

                if self.is_boundary_face(face_index, self.get_boundary_markers()):
                    boundary_marker = self.find_boundary_marker(face_index,
                                                                self.get_boundary_markers())
                    self.add_boundary_face(boundary_marker,
                                           new_face_index, cell_orientations[local_face_index])

                self.set_cell_orientation(cell_index,
                                          np.array(list(cell_orientations) +
                                                   [cell_orientations[local_face_index]]))

                cell_next_door = list(self.get_face_to_cell(face_index))
                cell_next_door.remove(cell_index)

                if len(cell_next_door) == 1:
                    next_door_faces = self.get_cell(cell_next_door[0])
                    next_door_local_face_index = list(next_door_faces).index(face_index)
                    next_door_faces = list(next_door_faces) + [new_face_index]
                    next_door_orientations = self.get_cell_normal_orientation(cell_next_door[0])
                    next_door_orientations = list(next_door_orientations) + \
                        [next_door_orientations[next_door_local_face_index]]
                    next_door_orientations = np.array(next_door_orientations)

                    self.set_cell_faces(cell_next_door[0], next_door_faces)
                    self.set_cell_orientation(cell_next_door[0],
                                              next_door_orientations)
                    if cell_next_door[0] in face_segments_to_be_added:
                        face_segments_to_be_added[cell_next_door[0]] += [interior_face_segments[-1]]
                    else:
                        face_segments_to_be_added[cell_next_door[0]] = [interior_face_segments[-1]]


        if cell_index in face_segments_to_be_added:
            interior_face_segments += face_segments_to_be_added[cell_index]

        if len(interior_face_segments) > 0:
            new_face = self.construct_polygon_from_segments(interior_face_segments)

        for i in range(1):
            v1 = self.get_point(new_face[i+1]) - self.get_point(new_face[i])
            v2 = self.get_point(new_face[i]) - self.get_point(new_face[i-1])
            new_face_normal = np.cross(v2, v1)

            new_face_normal /= np.linalg.norm(new_face_normal)

        new_face_index = self.add_face(new_face)

        (face_area, face_centroid) = self.find_face_centroid(new_face_index)
        self.set_face_real_centroid(new_face_index, face_centroid)
        self.set_face_area(new_face_index, face_area)

        new_face_normal = self.find_face_normal(new_face_index)
        self.set_face_normal(new_face_index, new_face_normal)

        faces_for_cell_1 = []
        faces_for_cell_2 = []

        normals_for_cell_1 = []
        normals_for_cell_2 = []

        for face_index in self.get_cell(cell_index):
            current_center = self.get_face_real_centroid(face_index)
            plane_to_center = point_on_plane - current_center

            if np.dot(plane_to_center, plane_normal) > 0.:
                faces_for_cell_1.append(face_index)
                local_face_index = list(self.get_cell(cell_index)).index(face_index)
                face_normal = self.get_cell_normal_orientation(cell_index)[local_face_index]
                normals_for_cell_1.append(face_normal)

            else:
                faces_for_cell_2.append(face_index)
                local_face_index = list(self.get_cell(cell_index)).index(face_index)
                face_normal = self.get_cell_normal_orientation(cell_index)[local_face_index]
                normals_for_cell_2.append(face_normal)

        faces_for_cell_1.append(new_face_index)
        faces_for_cell_2.append(new_face_index)

        if np.dot(new_face_normal, plane_normal)>0.:
            normals_for_cell_1.append(1)
            normals_for_cell_2.append(-1)

        else:
            normals_for_cell_1.append(-1)
            normals_for_cell_2.append(1)

        self.set_cell_faces(cell_index, faces_for_cell_1)
        self.set_cell_orientation(cell_index, normals_for_cell_1)

        (cell_volume, cell_centroid) = self.find_volume_centroid(cell_index)
        self.set_cell_real_centroid(cell_index, cell_centroid)
        self.set_cell_volume(cell_index, cell_volume)

        new_cell_index = self.add_cell(faces_for_cell_2,
                                       normals_for_cell_2)

        (cell_volume, cell_centroid) = self.find_volume_centroid(new_cell_index)

        self.set_cell_volume(new_cell_index, cell_volume)
        self.set_cell_real_centroid(new_cell_index, cell_centroid)

        self.set_cell_k(new_cell_index, self.get_cell_k(cell_index))

        return new_cell_index


    def build_frac_from_faces(self, faces):
        """ Takes a list of face indices, and
        extrudes them into cells.
        """
        connections = []
        non_connected_edges = []
        for face in faces:
            non_connected_edges.append([])
            for local_edge_index in range(len(self.get_face(face))):
                non_connected_edges[-1].append(local_edge_index)

        for local_face_index in range(len(faces)):
            current_face_points = list(self.get_face(faces[local_face_index]))
            current_face_points.append(current_face_points[0])
            for local_point_index in range(len(current_face_points)-1):
                point_1 = self.get_point(current_face_points[local_point_index])
                point_2 = self.get_point(current_face_points[local_point_index+1])
                for local_face_index_2 in range(local_face_index+1, len(faces)):
                    current_face_points_2 = list(self.get_face(faces[local_face_index_2]))
                    current_face_points_2.append(current_face_points_2[0])

                    for local_point_index_2 in range(len(current_face_points_2)-1):
                        point_1_2 = self.get_point(current_face_points_2[local_point_index_2])
                        point_2_2 = self.get_point(current_face_points_2[local_point_index_2+1])

                        if np.linalg.norm(abs(point_1-point_1_2)+abs(point_2-point_2_2))< 1.e-12:
                            connections.append([local_face_index,
                                                local_face_index_2,
                                                local_point_index,
                                                local_point_index+1,
                                                local_point_index_2,
                                                local_point_index_2+1,
                                                1])

                            if local_point_index in non_connected_edges[local_face_index]:
                                non_connected_edges[local_face_index].remove(local_point_index)
                            if local_point_index_2 in non_connected_edges[local_face_index_2]:
                                non_connected_edges[local_face_index_2].remove(local_point_index_2)

                        if np.linalg.norm(abs(point_2-point_1_2)+abs(point_1-point_2_2))< 1.e-12:
                            connections.append([local_face_index,
                                                local_face_index_2,
                                                local_point_index,
                                                local_point_index+1,
                                                local_point_index_2,
                                                local_point_index_2+1,
                                                0])

                            if local_point_index in non_connected_edges[local_face_index]:
                                non_connected_edges[local_face_index].remove(local_point_index)
                            if local_point_index_2 in non_connected_edges[local_face_index_2]:
                                non_connected_edges[local_face_index_2].remove(local_point_index_2)

        ##Find edges that have more than two connections
        multiple_connection_indices = []
        multiple_connection_groups = []

        for (connection_index, connection) in enumerate(connections):

            first_face = self.get_face(faces[connection[0]])
            point1 = self.get_point(first_face[connection[2]])
            point2 = self.get_point(first_face[connection[3]%len(first_face)])

            multiple_connection_groups.append([])
            for remote_connection_index in range(connection_index+1, len(connections)):
                if remote_connection_index not in multiple_connection_indices:
                    remote_connection = connections[remote_connection_index]

                    first_face_remote = self.get_face(faces[remote_connection[0]])
                    point1_remote = self.get_point(first_face_remote[remote_connection[2]])
                    point2_remote = self.get_point(first_face_remote[remote_connection[3]%len(first_face_remote)])
                    if np.linalg.norm(abs(point1-point1_remote)+abs(point2-point2_remote))< 1.e-10 or\
                         np.linalg.norm(abs(point2-point1_remote)+abs(point1-point2_remote))< 1.e-10:
                        if len(multiple_connection_groups[-1]) == 0:
                            multiple_connection_groups[-1].append(connection_index)
                            multiple_connection_groups[-1].append(remote_connection_index)
                            multiple_connection_indices.append(connection_index)
                            multiple_connection_indices.append(remote_connection_index)
                        else:
                            multiple_connection_groups[-1].append(remote_connection_index)
                            multiple_connection_indices.append(remote_connection_index)

        multiple_connection_groups = [x for x in multiple_connection_groups if len(x)>0]

        new_multiple_connection_groups = []
        ## Switch from connection index to actual connections
        for group in multiple_connection_groups:
            new_multiple_connection_groups.append([])
            for connection_index in group:
                new_multiple_connection_groups[-1].append(list(connections[connection_index]))

        multiple_connection_groups = new_multiple_connection_groups
        connections_without_multiple = []

        for connection_index in range(len(connections)):
            if connection_index not in multiple_connection_indices:
                connections_without_multiple.append(connections[connection_index])

        subface_connections = []
        ## For multiple connections, the joining faces must be divided
        ## to two, a top and a bottom face.
        for group in multiple_connection_groups:
            done_faces = []
            for connection in group:
                current_face = connection[0]
                full_face = self.get_face(faces[current_face])
                point1 = self.get_point(full_face[connection[2]])
                point2 = self.get_point(full_face[connection[3]%len(full_face)])
                current_centroid = self.get_face_real_centroid(faces[current_face])
                home_vector = current_centroid-(point1+point2)/2.
                home_vector /= np.linalg.norm(home_vector)
                norm1 = self.get_face_normal(faces[current_face])
                max_top_angle = -999
                max_top_connection_index =  None
                max_bot_angle = -999
                max_bot_connection_index = None
                for (connection_index, connection2) in enumerate(group):
                    if connection2[0] == current_face:
                        full_face_2 = self.get_face(faces[connection2[1]])
                        point1_2 = self.get_point(full_face_2[connection2[4]])
                        point2_2 = self.get_point(full_face_2[connection2[5]%len(full_face_2)])
                        centroid_2 = self.get_face_real_centroid(faces[connection2[1]])
                        current_vector = centroid_2-(point1_2+point2_2)/2.
                        current_vector /= np.linalg.norm(current_vector)
                        top_angle = np.dot(home_vector, current_vector)
                        if np.dot(current_vector, norm1) > 0.:
                            bottom_angle = -top_angle
                            top_angle += 2
                        else:
                            bottom_angle = top_angle+2
                            top_angle = -top_angle

                        if top_angle > max_top_angle:
                            max_top_angle = top_angle
                            max_top_connection_index = connection_index

                        if bottom_angle > max_bot_angle:
                            max_bot_angle = bottom_angle
                            max_bot_connection_index = connection_index

                face2_top = group[max_top_connection_index][1]
                face2_bot = group[max_bot_connection_index][1]

                if ((current_face, 'TOP')) not in done_faces:
                    subface_connections.append(group[max_top_connection_index]+['TOP'])
                    done_faces.append((current_face, 'TOP'))
                    if group[max_top_connection_index][6] == 0:
                        done_faces.append((face2_top, 'TOP'))
                    else:
                        done_faces.append((face2_top, 'BOT'))

                if ((current_face, 'BOT')) not in done_faces:
                    subface_connections.append(group[max_bot_connection_index]+['BOT'])
                    done_faces.append((current_face, 'BOT'))
                    if group[max_bot_connection_index][6] == 0:
                        done_faces.append((face2_bot, 'BOT'))
                    else:
                        done_faces.append((face2_bot, 'TOP'))

        connections = connections_without_multiple

        ## Loop through all the connections, and find out which
        ## faces are to contribute to the normal calculation.
        for connection in connections:
            ## The two faces already there
            face_1 = connection[0]
            face_2 = connection[1]
            faces_for_point_1 = set([(connection[0], 0),
                                     (connection[1], connection[6])])
            faces_for_point_2 = set([(connection[0], 0),
                                     (connection[1], connection[6])])
            ## Loop through all the connections, and find overlap
            ## between the local point indices.
            for remote_connection in connections:
                if remote_connection[0] == face_1 and \
                        remote_connection[1] == face_2:
                    pass
                else:

                    if remote_connection[0] == face_1:
                        if connection[2] in remote_connection[2:4]:
                            faces_for_point_1.add((remote_connection[1],
                                                   remote_connection[6]))

                        if connection[3] in remote_connection[2:4]:
                            faces_for_point_2.add((remote_connection[1],
                                                   remote_connection[6]))

                    if remote_connection[1] == face_1:
                        if connection[2] in remote_connection[4:6]:
                            faces_for_point_1.add((remote_connection[0],
                                                   remote_connection[6]))

                        if connection[3] in remote_connection[4:6]:
                            faces_for_point_2.add((remote_connection[0],
                                                   remote_connection[6]))

                    if remote_connection[0] == face_2:
                        if connection[4] in remote_connection[2:4]:
                            if connection[6] == 1:
                                faces_for_point_1.add((remote_connection[1],
                                                       remote_connection[6]^connection[6]))
                            else:
                                faces_for_point_2.add((remote_connection[1],
                                                       remote_connection[6]^connection[6]))

                        if connection[5] in remote_connection[2:4]:
                            if connection[6] == 1:
                                faces_for_point_2.add((remote_connection[0],
                                                       remote_connection[6]^connection[6]))
                            else:
                                faces_for_point_1.add((remote_connection[0],
                                                       remote_connection[6]^connection[6]))

                    if remote_connection[1] == face_2:
                        if connection[4] in remote_connection[4:6]:
                            if connection[6] == 1:
                                faces_for_point_1.add((remote_connection[1],
                                                       remote_connection[6]^connection[6]))
                            else:
                                faces_for_point_2.add((remote_connection[1],
                                                       remote_connection[6]^connection[6]))

                        if connection[5] in remote_connection[4:6]:
                            if connection[6] == 1:
                                faces_for_point_2.add((remote_connection[0],
                                                       remote_connection[6]^connection[6]))
                            else:
                                faces_for_point_1.add((remote_connection[0],
                                                       remote_connection[6]^connection[6]))

            connection += [faces_for_point_1, faces_for_point_2]

        ## Loop through all the connections, and find out which
        ## faces are to contribute to the normal calculation.
        for connection in subface_connections:
            ## The two faces already there
            face_1 = connection[0]
            face_2 = connection[1]
            faces_for_point_1 = set([(connection[0], 0),
                                     (connection[1], connection[6])])
            faces_for_point_2 = set([(connection[0], 0),
                                     (connection[1], connection[6])])

            connection += [faces_for_point_1, faces_for_point_2]


        # Build the faces.
        new_faces = []
        ## Maps the cells to the walls that make them.
        ## The cells are identified by the index of the
        ## generating face.
        face_to_walls = {}
        top_points = []
        bot_points = []
        mid_points = []
        for face in range(len(faces)):
            face_to_walls[face] = []
            top_points.append([-1]*len(self.get_face(faces[face])))
            bot_points.append([-1]*len(self.get_face(faces[face])))
            mid_points.append([-1]*len(self.get_face(faces[face])))

        for connection in connections:
            norm1 = np.zeros(3)
            for (local_face_index, orientation) in connection[7]:
                if orientation == 0:
                    norm1 += self.get_face_normal(faces[local_face_index])
                else:
                    norm1 -= self.get_face_normal(faces[local_face_index])

            norm2 = np.zeros(3)
            for (local_face_index, orientation) in connection[8]:
                if orientation == 0:
                    norm2 += self.get_face_normal(faces[local_face_index])
                else:
                    norm2 -= self.get_face_normal(faces[local_face_index])

            norm1 /= np.linalg.norm(norm1)
            norm2 /= np.linalg.norm(norm2)

            width = .005
            full_face = self.get_face(faces[connection[0]])
            point1 = -width*norm1+self.get_point(full_face[connection[2]])
            point2 = -width*norm2+self.get_point(full_face[connection[3]%len(full_face)])
            point3 = width*norm2+self.get_point(full_face[connection[3]%len(full_face)])
            point4 = width*norm1+self.get_point(full_face[connection[2]])

            point_1_index = self.add_point(point1)
            point_2_index = self.add_point(point2)
            point_3_index = self.add_point(point3)
            point_4_index = self.add_point(point4)

            new_face_index = self.add_face([point_1_index,
                                            point_2_index,
                                            point_3_index,
                                            point_4_index,])

            (area, centroid) =  self.find_face_centroid(new_face_index)
            current_face_normal = self.find_face_normal(new_face_index)
            self.set_face_normal(new_face_index, current_face_normal)

            self.set_face_real_centroid(new_face_index, centroid)

            if self.has_face_shifted_centroid:
                self.set_face_shifted_centroid(new_face_index, centroid)

            self.set_face_area(new_face_index, area)

            face_to_walls[connection[0]].append((new_face_index, 1))
            face_to_walls[connection[1]].append((new_face_index, -1))

            top_points[connection[0]][connection[2]] = point_4_index
            top_points[connection[0]][connection[3]%
                                      len(self.get_face(faces[connection[0]]))]=point_3_index

            bot_points[connection[0]][connection[2]]=point_1_index
            bot_points[connection[0]][connection[3]%
                                      len(self.get_face(faces[connection[0]]))]=point_2_index

            if connection[6] == 0:
                top_points[connection[1]][connection[5]%
                                      len(self.get_face(faces[connection[1]]))]=point_4_index
                top_points[connection[1]][connection[4]]=point_3_index

                bot_points[connection[1]][connection[5]%
                                          len(self.get_face(faces[connection[1]]))]=point_1_index
                bot_points[connection[1]][connection[4]]=point_2_index

            else:
                bot_points[connection[1]][connection[4]]=point_4_index
                bot_points[connection[1]][connection[5]%
                                      len(self.get_face(faces[connection[1]]))]=point_3_index

                top_points[connection[1]][connection[4]]=point_1_index
                top_points[connection[1]][connection[5]%
                                      len(self.get_face(faces[connection[1]]))]=point_2_index

            new_faces.append(new_face_index)


        ## Build the subfaces.
        for connection in subface_connections:
            norm1 = np.zeros(3)
            for (local_face_index, orientation) in connection[8]:
                if orientation == 0:
                    norm1 += self.get_face_normal(faces[local_face_index])
                else:
                    norm1 -= self.get_face_normal(faces[local_face_index])

            norm2 = np.zeros(3)
            for (local_face_index, orientation) in connection[8]:
                if orientation == 0:
                    norm2 += self.get_face_normal(faces[local_face_index])
                else:
                    norm2 -= self.get_face_normal(faces[local_face_index])

            norm1 /= np.linalg.norm(norm1)
            norm2 /= np.linalg.norm(norm2)

            width = .007

            if connection[7] == 'TOP':
                full_face = self.get_face(faces[connection[0]])
                point1 = self.get_point(full_face[connection[2]])
                point2 = self.get_point(full_face[connection[3]%len(full_face)])
                point3 = width*norm2+self.get_point(full_face[connection[3]%len(full_face)])
                point4 = width*norm1+self.get_point(full_face[connection[2]])

            elif connection[7] == 'BOT':
                full_face = self.get_face(faces[connection[0]])
                point1 = -width*norm1+self.get_point(self.get_face(faces[connection[0]])[connection[2]])
                point2 = -width*norm2+self.get_point(full_face[connection[3]%len(full_face)])
                point3 = self.get_point(full_face[connection[3]%len(full_face)])
                point4 = self.get_point(full_face[connection[2]])

            point_1_index = self.add_point(point1)
            point_2_index = self.add_point(point2)
            point_3_index = self.add_point(point3)
            point_4_index = self.add_point(point4)

            new_face_index = self.add_face([point_1_index,
                                            point_2_index,
                                            point_3_index,
                                            point_4_index,])

            (area, centroid) =  self.find_face_centroid(new_face_index)
            current_face_normal = self.find_face_normal(new_face_index)
            self.set_face_normal(new_face_index, current_face_normal)
            self.set_face_real_centroid(new_face_index, centroid)

            if self.has_face_shifted_centroid:
                self.set_face_shifted_centroid(new_face_index, centroid)

            self.set_face_area(new_face_index, area)

            face_to_walls[connection[0]].append((new_face_index, 1))
            face_to_walls[connection[1]].append((new_face_index, -1))

            if connection[7] == 'TOP':
                mid_points[connection[0]][connection[2]]=point_1_index
                mid_points[connection[0]][connection[3]%
                                          len(self.get_face(faces[connection[0]]))]=point_2_index

                top_points[connection[0]][connection[3]%
                                          len(self.get_face(faces[connection[0]]))]=point_3_index
                top_points[connection[0]][connection[2]]=point_4_index


            elif connection[7] == 'BOT':
                bot_points[connection[0]][connection[2]]=point_1_index
                bot_points[connection[0]][connection[3]%
                                          len(self.get_face(faces[connection[0]]))]=point_2_index

                mid_points[connection[0]][connection[2]]=point_4_index
                mid_points[connection[0]][connection[3]%
                                          len(self.get_face(faces[connection[0]]))]=point_3_index

            if connection[7] == 'TOP':
                if connection[6] == 0:
                    top_points[connection[1]][connection[5]%
                                              len(self.get_face(faces[connection[1]]))]=point_4_index
                    top_points[connection[1]][connection[4]%
                                              len(self.get_face(faces[connection[1]]))]=point_3_index

                    mid_points[connection[1]][connection[5]%
                                              len(self.get_face(faces[connection[1]]))]=point_1_index
                    mid_points[connection[1]][connection[4]%
                                              len(self.get_face(faces[connection[1]]))]=point_2_index

                else:
                    bot_points[connection[1]][connection[4]]=point_4_index
                    bot_points[connection[1]][connection[5]%
                                              len(self.get_face(faces[connection[1]]))]=point_3_index

                    mid_points[connection[1]][connection[4]]=point_1_index
                    mid_points[connection[1]][connection[5]%
                                              len(self.get_face(faces[connection[1]]))]=point_2_index

            if connection[7] == 'BOT':
                if connection[6] == 0:
                    mid_points[connection[1]][connection[5]%
                                              len(self.get_face(faces[connection[1]]))]=point_4_index
                    mid_points[connection[1]][connection[4]]=point_3_index

                    bot_points[connection[1]][connection[5]%
                                              len(self.get_face(faces[connection[1]]))]=point_1_index
                    bot_points[connection[1]][connection[4]]=point_2_index

                else:
                    mid_points[connection[1]][connection[4]]=point_4_index
                    mid_points[connection[1]][connection[5]%
                                              len(self.get_face(faces[connection[1]]))]=point_3_index

                    top_points[connection[1]][connection[4]%
                                              len(self.get_face(faces[connection[1]]))]=point_1_index
                    top_points[connection[1]][connection[5]%
                                              len(self.get_face(faces[connection[1]]))]=point_2_index


            new_faces.append(new_face_index)

        for (local_face_index, face) in enumerate(non_connected_edges):
            for point1 in face:
                global_face_index = faces[local_face_index]
                point2 = (point1+1)%len(self.get_face(global_face_index))

                new_face_points = []
                norm = self.get_face_normal(global_face_index)
                width = .005
                if bot_points[local_face_index][point1] == -1:
                    new_point = -width*norm+self.get_point(self.get_face(global_face_index)[point1])
                    point_1_index = self.add_point(new_point)
                    bot_points[local_face_index][point1] = point_1_index
                else:
                    point_1_index = bot_points[local_face_index][point1]

                new_face_points.append(point_1_index)

                if bot_points[local_face_index][point2] == -1:
                    new_point = -width*norm+self.get_point(self.get_face(global_face_index)[point2])
                    point_2_index = self.add_point(new_point)
                    bot_points[local_face_index][point2] = point_2_index
                else:
                    point_2_index = bot_points[local_face_index][point2]

                new_face_points.append(point_2_index)

                if mid_points[local_face_index][point2] != -1:
                    new_face_points.append(mid_points[local_face_index][point2])

                if top_points[local_face_index][point2] == -1:
                    new_point = width*norm+self.get_point(self.get_face(global_face_index)[point2])
                    point_3_index = self.add_point(new_point)
                    top_points[local_face_index][point2] = point_3_index
                else:
                    point_3_index = top_points[local_face_index][point2]

                new_face_points.append(point_3_index)

                if top_points[local_face_index][point1] == -1:
                    new_point = width*norm+self.get_point(self.get_face(global_face_index)[point1])
                    point_4_index = self.add_point(new_point)
                    top_points[local_face_index][point1] = point_4_index
                else:
                    point_4_index = top_points[local_face_index][point1]

                new_face_points.append(point_4_index)

                if mid_points[local_face_index][point1] != -1:
                    new_face_points.append(mid_points[local_face_index][point1])

                new_face_index = self.add_face(new_face_points)
                current_face_normal = self.find_face_normal(new_face_index)
                self.set_face_normal(new_face_index, current_face_normal)

                (area, centroid) =  self.find_face_centroid(new_face_index)
                self.set_face_real_centroid(new_face_index, centroid)

                if self.has_face_shifted_centroid:
                    self.set_face_shifted_centroid(new_face_index, centroid)

                self.set_face_area(new_face_index, area)

                if np.dot(current_face_normal, centroid -self.get_face_real_centroid(global_face_index))>  0.:
                    self.add_internal_no_flow(new_face_index, 1)
                else:
                    self.add_internal_no_flow(new_face_index, -1)

                face_to_walls[local_face_index].append((new_face_index, 1))

                new_faces.append(new_face_index)


        ## Adds the top and bottom faces.
        for local_face_index in range(len(faces)):
            new_face_points = top_points[local_face_index]
            new_face_index = self.add_face(new_face_points)
            face_to_walls[local_face_index].append((new_face_index, 1))
            self.add_internal_no_flow(new_face_index, 1)

            (area, centroid) =  self.find_face_centroid(new_face_index)
            current_face_normal = self.find_face_normal(new_face_index)
            self.set_face_normal(new_face_index, current_face_normal)
            self.set_face_real_centroid(new_face_index, centroid)

            if self.has_face_shifted_centroid:
                self.set_face_shifted_centroid(new_face_index, centroid)

            self.set_face_area(new_face_index, area)

            new_faces.append(new_face_index)

            new_face_points = bot_points[local_face_index]
            new_face_index = self.add_face(new_face_points)
            face_to_walls[local_face_index].append((new_face_index, -1))

            self.add_internal_no_flow(new_face_index, 1)

            (area, centroid) =  self.find_face_centroid(new_face_index)
            current_face_normal = self.find_face_normal(new_face_index)
            self.set_face_normal(new_face_index, current_face_normal)

            self.set_face_real_centroid(new_face_index, centroid)

            if self.has_face_shifted_centroid:
                self.set_face_shifted_centroid(new_face_index, centroid)

            self.set_face_area(new_face_index, area)

            new_faces.append(new_face_index)

        # Duplicate reservoir face for interior dirichlet boundary.
        for face in face_to_walls:
            top_res_face_index = faces[face]

            if self.face_to_cell[top_res_face_index, 0] >= 0 and \
                    self.face_to_cell[top_res_face_index, 1] >= 0 :
                bot_res_face_index = self.add_face(list(self.get_face(top_res_face_index)))
                self.set_face_area(bot_res_face_index, self.get_face_area(top_res_face_index))
                self.set_face_normal(bot_res_face_index, self.get_face_normal(top_res_face_index))
                self.set_face_real_centroid(bot_res_face_index,
                                            self.get_face_real_centroid(top_res_face_index))
                if self.has_face_shifted_centroid:
                    self.set_face_shifted_centroid(bot_res_face_index,
                                                   self.get_face_real_centroid(top_res_face_index))

                bottom_cell = self.face_to_cell[top_res_face_index, 1]

                new_cell_faces = array.array('i', self.get_cell(bottom_cell))
                local_face_index_in_cell = list(new_cell_faces).index(top_res_face_index)

                new_cell_faces[local_face_index_in_cell] = bot_res_face_index

                top_cell_index = self.face_to_cell[top_res_face_index, 0]
                local_top_face_index_in_cell = list(self.get_cell(top_cell_index)).index(top_res_face_index)
                top_res_face_orientation =\
                    self.get_cell_normal_orientation(top_cell_index)[local_top_face_index_in_cell]

                self.remove_from_face_to_cell(top_res_face_index, bottom_cell)
                self.set_cell_faces(bottom_cell, new_cell_faces)
            else:
                raise Exception("Face on boundary encountered")

            new_cell_index = self.add_cell(array.array('i', [x[0] for x in face_to_walls[face]]),
                                           array.array('i', [x[1] for x in face_to_walls[face]]))
            self.set_cell_domain(new_cell_index, 1)

            (volume, centroid) = self.find_volume_centroid(new_cell_index)
            self.set_cell_volume(new_cell_index, volume)
            self.set_cell_real_centroid(new_cell_index, centroid)
            if self.has_cell_shifted_centroid:
                self.set_cell_shifted_centroid(new_cell_index, centroid)

            self.set_cell_k(new_cell_index, np.eye(3)*1.)

            self.set_forcing_pointer(new_cell_index,
                                     [top_res_face_index, bot_res_face_index],
                                     [top_res_face_orientation, -top_res_face_orientation])

            self.set_dirichlet_face_pointer(top_res_face_index,
                                            top_res_face_orientation,
                                            new_cell_index)
            self.set_dirichlet_face_pointer(bot_res_face_index,
                                            -top_res_face_orientation,
                                            new_cell_index)

        self.output_vtk_faces("new_faces", new_faces)

    def build_mesh(self):
        """ Base class function for constructing the mesh.
        """
        raise NotImplementedError

