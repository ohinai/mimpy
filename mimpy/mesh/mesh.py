"""
mesh module. 
"""
import numpy as np
import os 
import array
import math

try:
    import matplotlib
    from matplotlib.patches import Circle, Wedge, Polygon
    from matplotlib.collections import PatchCollection

    import pylab
except ImportError:
    print "matplotlib not installed."

class Mesh:
    """ The *Mesh* class is the basic intereface for accessing 
    the data needed from a mesh to build an MFD discretization. 
    When correctly populated, an instance of Mesh
    would lead to a correctly constructed saddle-point
    system using the MFD class. It stores basic information 
    about the geometry of the mesh as well cell and face 
    properties. The class is generic enough to represent 
    both 2 and 3 dimensional meshes. The class also 
    contains basic functions for plotting and 
    visualization. 

    The Mesh class is not intended to be used on its own, 
    rather it should be subclassed for a specific type of 
    mesh. For example, when building a 3 dimensional 
    hexahedral mesh, a sublcass HexMesh inherits
    from Mesh, and adds the appropriate functionality. 
    """
    def __init__(self):
        # List of points used to construct mesh faces. 
        # Each point coordinate is prepresented by 
        # a Numpy array. 
        self.points = []

        # List of mesh faces, each face is represented by the 
        # a list of points. In 2D, it's a list of pairs of poitns. 
        # In 3D, it's an ordered list of points that make up the 
        # polygon.         
        self.faces = []

        # Face normals. 
        self.face_normals = []

        # Area of mesh face. 
        self.face_areas = []

        # The centroid of face. 
        self.face_real_centroids = []

        # Dict that maps faces to the cells 
        # they are in. 
        self.face_to_cell = {}
        
        # Dict from cell to a list of faces 
        # in the cell that are also neumann faces. 
        # Used for incroporating the boundary 
        # conditions when building m and div_t. 
        self.cell_faces_neumann = {}

        # A point on the plane of the face that is used 
        # to build the MFD matrix R. This point does 
        # not have to be on the face itself. 
        self.face_shifted_centroids = []

        self.has_face_shifted_centroid = False
        self.has_cell_shifted_centroid = False
        
        self.has_two_d_polygons = False
        self.has_alpha = False

        self.face_quadrature_points = {}
        self.face_quadrature_weights = {}

        self.boundary_markers = []
        self.boundary_descriptions = []

        # Hash from face marker => [[face index, face normal orientation]] 
        self.boundary_faces = {}
        
        # List of cells. Each cell is made up of a list
        # of faces.         
        self.cells = []

        # For each cell, a list of bools indicating 
        # whether the normal in self.face_normals
        # is in or out of the cell. 
        self.cell_normal_orientation  = []

        # Cell Volumes. 
        self.cell_volume = []

        # List of cell centroids. 
        self.cell_real_centroid = []

        # Points used inside the cell used
        # to build the MFD matrix R. 
        self.cell_shifted_centroid = []
        
        self.cell_quadrature_points = {}
        self.cell_quadrature_weights = {}
        
        self.cell_k = []
        self.cell_k_inv = []

        # Tags cells depending on which domain 
        # they belong to (for fractures and 
        # multi-domain problems)
        self.cell_domain = []

        # Dimesion of the mesh. 
        self.dim = 2

        # For 2D meshes, optional polygon representation of cells. 
        self.two_d_polygons = []

        # dict: {face_index: Dirichlet value, ... }
        self.dirichlet_boundary_values = {}

        # dict: {face_index: Neumann value, ... }
        self.neumann_boundary_values = {}

        # dict: {face_index: (cell_index, face_orientation), ...}
        # Allows Dirichlet boundaries to be set implicitly 
        # based on pressure of cells. 
        self.dirichlet_boundary_pointers = {}

        # dict: {face_index: [face_index_1, face_index_2, ...], ...}
        # Allows Neumann boundaries to be set implicitly 
        # based on fluxes at other faces. 
        self.neumann_boundary_pointers = {}

        # dict: {face_index: (lagrange_index, orientation)}
        # Allows dirichlet boundaries to point to 
        # lagrange multipliers for domain decomposition. 
        # A lagrange multiplier is a face identicial 
        # to the one pointing to it, but not associated 
        # with any cells.  
        self.face_to_lagrange_pointer = {}

        # dict: lagrange_index: [(face_index_1, orientation), ...], ...}
        # Points a lagrange multiplier to faces associated 
        # with it. Is treated like a forcing function. 
        self.lagrange_to_face_pointers = {}

        # dict: {cell_index: [(face_index_1, orientation), ...], ...}
        # Allows source terms to be set implicitly
        # based on fluxes at other faces. 
        self.forcing_function_pointers = {}

        # list: [F1, F2, F3, ....]
        self.cell_forcing_function = []

        # Lowest order term coef
        # List: [alpha1, alpha2, ...]
        self.cell_alpha = []
        self.is_using_alpha_list = False
        
        # Rate wells cell location
        self.rate_wells = []
        self.rate_wells_rate = []
        self.rate_wells_name = []

        self.gravity_vector = None
        self.fluid_density = None
        self.gravity_acceleration = None
        
        self.use_gravity = False

    def add_point(self, new_point):
        """ Takes a Numpy array 
        representing the cartesian point coodrinates, 
        and appends the point to the end of the point list. 
        Returns the index of the new point.
        """
        self.points.append(new_point)
        return len(self.points)-1
    
    def get_point(self, point_index):
        """ Takes a point index and returns 
        a Numpy array of point coodrinates. 
        """
        return self.points[point_index]
    
    def get_number_of_points(self):
        """ Returns the total number of points. 
        """
        return len(self.points)

    def add_face(self, list_of_points):
        """ Takes a list of point indices, and 
        appends them to the list of faces 
        in the mesh. The point indices must 
        be oriented in a clockwise direction 
        relative to the face normal. In 2D, a 
        face is represnted by two points. 
        Returns the index of the new face. 
        """
        self.faces.append(list_of_points)
        self.face_normals.append(None)
        self.face_areas.append(None)
        self.face_real_centroids.append(None)

        self.face_to_cell[len(self.faces)-1] = []

        if self.has_face_shifted_centroid:
            self.face_shifted_centroids.append(None)

        return len(self.faces)-1

    def set_face(self, face_index, points):
        """ Sets a new set of points for a given face_index.
        """
        self.faces[face_index] = points

    def duplicate_face(self, face_index):
        """ Creates new face with all the properties 
        of the face_index, and adds the face to the 
        bottom of the face list. The function 
        returns the new face index. 
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
        """
        return self.faces[face_index]

    def get_number_of_face_points(self, face_index):
        """ Returns the number of points that make 
        up a given face. 
        """
        return len(self.faces[face_index])

    def get_number_of_faces(self):
        """ Returns the total number of faces 
        in the mesh. This corresponds to the 
        number of velocity degrees of freedom.  
        """
        return len(self.faces)
    
    def get_number_of_cell_faces(self, cell_index):
        """ Returns the number of faces for cell_index
        """
        return len(self.cells[cell_index])

    def get_cell_faces_neumann(self, cell_index):
        """ Returns all the faces in cell_index that 
        are also neumann faces. 
        """
        if self.cell_faces_neumann.has_key(cell_index):
            return self.cell_faces_neumann[cell_index]
        else:
            return []

    def is_line_seg_intersect_face(self, face_index, p1, p2):
        """ Returns True if the line segment 
        intersects with a face. 
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

            if d<= length+1.e-8 and  d> 0.+1.e-8:
                intersection_point = d*vector+p1
                
                direction = np.zeros(len(self.get_face(face_index)))

                normal = self.get_face_normal(face_index)
                current_point = self.get_point(self.get_face(face_index)[-1])
                for (local_index, next_point_index) in enumerate(self.get_face(face_index)):
                    next_point = self.get_point(next_point_index)
                    
                    face_vec = next_point - current_point
                    check_vec = current_point - intersection_point
                    
                    direction[local_index] = np.dot(np.cross(face_vec, check_vec), normal)
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
        for cell_index in range(number_of_cells):
            self.cells.append([])
            self.cell_normal_orientation.append([])
            self.cell_forcing_function.append(0.)
            self.cell_volume.append(None)
            self.cell_k.append(None)
            self.cell_domain.append(None)
            self.cell_real_centroid.append(None)
            if self.has_two_d_polygons:
                self.two_d_polygons.append(None)
            if self.has_alpha:
                self.cell_alpha.append(None)

    def set_cell_faces(self, cell_index, faces):
        """ Sets the cell faces. 
        """
        self.cells[cell_index] = faces
        for face_index in faces:
            if cell_index not in self.face_to_cell[face_index]:
                self.face_to_cell[face_index].append(cell_index)

    def set_cell_orientation(self, cell_index, orientation):
        """ Sets the cell orientation. 
        """
        self.cell_normal_orientation[cell_index] = orientation
        
    def add_cell(self, 
                 list_of_faces, 
                 list_of_orientations, 
                 forcing_function = 0., 
                 quadrature_points = None, 
                 quadrature_weights = None, 
                 cell_volume = None, 
                 cell_k = None, cell_domain = 0):
        """ Adds a new cell to the mesh. A cell is represented 
        by a list of face indices. The function also 
        takes in a list of orientations of the same length
        as the list_of_faces. These represent the direction 
        of the face normal relative to the cell: 1 means it 
        points out, -1 means it points in to the cell. 
        Returns the index of the new cell. 
        """
        self.cells.append(list_of_faces)
        self.cell_normal_orientation.append(list_of_orientations)
        self.cell_forcing_function.append(forcing_function)
        self.cell_volume.append(cell_volume)
        self.cell_k.append(cell_k)

        for face_index in list_of_faces:
            self.face_to_cell[face_index].append(len(self.cells)-1)

        self.cell_domain.append(cell_domain)

        self.cell_real_centroid.append(None)

        if self.has_two_d_polygons:
            self.two_d_polygons.append(None)
            
        if self.has_alpha:
            self.cell_alpha.append(None)
        
        if self.has_cell_shifted_centroid: 
            self.cell_shifted_centroid.append(None)
        
        return len(self.cells)-1

    def get_cell(self, cell_index):
        """ Given a cell_index, it returns the list of faces 
        that make up that cell. 
        """
        return self.cells[cell_index]

    def get_number_of_cells(self):
        """ Returns total number of cells in mesh. 
        """
        return len(self.cells)
    
    def get_cell_normal_orientation(self, cell_index):
        """ Given a cell index, returns a list of face
        orientations for that cell.
        """
        return self.cell_normal_orientation[cell_index]

    def set_cell_real_centroid(self, cell_index, centroid):
        """ Sets the array of the cell centroid. 
        """
        self.cell_real_centroid[cell_index] = centroid
    
    def get_cell_real_centroid(self, cell_index):
        """ Returns array of the cell centroid
        """
        return self.cell_real_centroid[cell_index]

    def get_all_cell_real_centroids(self):
        """ Returns list of all cell centroids.
        """
        return self.cell_real_centroid

    def get_all_cell_shifted_centroids(self):
        """ Returns list of all cell centroids.
        """
        return self.cell_shifted_centroid

    def initialize_cell_shifted_centroid(self, number_of_cells = None):
        """ Informs Mesh that the shifted cell centroid data structure 
        will be utilized. This is necessary since shifted cell
        centroids are an optional data structure. 
        If already known, the number of cells can be initialized 
        using number_of_cells. The add_cell method 
        will add an extra entry for the shifted centroids. 
        """
        if number_of_cells is None:
            self.cell_shifted_centroid = [None]*self.get_number_of_cells()
        else:
            self.cell_shifted_centroid = [None]*number_of_cells

    def initialize_face_shifted_centroid(self, number_of_faces = None):
        """ Informs Mesh that the shifted face centroid data structure 
        will be utilized. This is necessary since shifted face
        centroids are an optional data structure. 
        If already known, the number of face can be initialized 
        using number_of_faces. The add_face method 
        will add an extra entry for the shifted centroids. 
        """
        if number_of_faces is None:
            self.face_shifted_centroids = [None]*self.get_number_of_faces()
        else:
            self.face_shifted_centroids = [None]*number_of_faces

    def set_cell_shifted_centroid(self, cell_index, centroid):
        """ Sets the shifted centroid. Since the shifted centroid 
        is a optional, 
        """
        self.cell_shifted_centroid[cell_index] = centroid

    def use_face_shifted_centroid(self):
        """ Informs the MFD class to use the shifted centriod 
        rather than the real face centroid. 
        """
        self.has_face_shifted_centroid = True

    def is_using_face_shifted_centroid(self):
        """ Returns True if MFD is to use the shifted 
        face centroids, False otherwise. 
        """
        return self.has_face_shifted_centroid

    def use_cell_shifted_centroid(self):
        """ Informs the MFD class to use the shifted centriod 
        rather than the real cell centroid. 
        """
        self.has_cell_shifted_centroid = True
    
    def is_using_cell_shifted_centroid(self):
        """ Returns True if MFD is to use the shifted 
        cell centroids, False otherwise. 
        """
        return self.has_cell_shifted_centroid

    def get_cell_shifted_centroid(self, cell_index):
        """ Returns the shifted cell centroid for cell_index
        """
        return self.cell_shifted_centroid[cell_index]
    
    def set_cell_volume(self, cell_index, volume):
        """ Sets cell volume for cell_index. 
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
        self.cell_k[cell_index] = k

    def get_cell_k(self, cell_index):
        """ Return permeability tensor k 
        (Numpy matrix) for cell_index.  
        """
        return self.cell_k[cell_index]

    def get_all_k(self):
        """ Returns a list of all cell 
        permeability tensors. 
        """
        return self.cell_k

    def use_two_d_polygons(self):
        """ This function must be called in order to 
        set polygon representations 
        of cells in 2D meshes. 
        
        2D polygons represents the 
        cell as set of the point indices 
        that make up a cell. This data structure 
        is useful when computing the area of the cell 
        as well as for visualization purposes. 
        However, it is no necessary for construction 
        of in the MFD class. 
        """        
        self.has_two_d_polygons = True
        
    def set_two_d_polygon(self, cell_index, polygon):
        """ Set the 2D polygon for cell_index. 
        polygon = [p_1, p_2, ... p_n]
        """
        self.two_d_polygons[cell_index] = polygon
        
    def get_2d_polygon(self, cell_index):
        """ Returns the 2D polygon for 
        cell_index. 
        """
        return self.two_d_polygons[cell_index]

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
    
    def set_cell_forcing_function(self, cell_index, forcing_value):
        """ Sets total forcing q (float) for cell_index        
        """
        self.cell_forcing_function[cell_index] = forcing_value

    def get_cell_forcing_function(self, cell_index):
        """ Returns total forcing (float) for cell_index. 
        """
        return self.cell_forcing_function[cell_index]

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
        the face_orientation indicated whether the normal of that
        face points in (-1) or out (1) of the cell the face
        belongs to. 
        
        A face should never be associated with more than one marker. 
        """
        self.boundary_faces[boundary_marker].append([face_index, 
                                                     face_orientation])

    def set_boundary_faces(self, boundary_marker, 
                           face_orientation_list):
        """ Takes a boundary_marker index, and sets the entire list 
        of tuples for that boundary marker. 
        """
        self.boundary_faces[boundary_marker] = face_orientation_list
        
    def get_boundary_faces_by_marker(self, boundary_marker):
        """ Returns a list of all the faces associated with a boundary_marker. 
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
            number_of_boundary_faces += len(self.boundary_faces[boundary_marker])

        return number_of_boundary_faces

    def set_face_quadrature_points(self, face_index, points):
        """ Sets quadrature points for face_index. The points are 
        a list of coordinates (numpy arrays). The points 
        are used to compute boundary integrals for 
        computing Neumann and Dirichlet boundary conditions 
        from functional forms. In addition, it can be 
        used for calculating velocity error against 
        a known solution. 
        """
        self.face_quadrature_points[face_index] = points

    def get_face_quadrature_points(self, face_index):
        """ Returns the quadrature points for face_index. 
        """
        return self.face_quadrature_points[face_index]

    def set_face_quadrature_weights(self, face_index, weights):
        """ Sets the weights associated with the corresponding 
        quadrature points for face_index. 
        weights = list of floats. 
        """
        self.face_quadrature_weights[face_index] = weights

    def get_face_quadrature_weights(self, face_index):
        """ Returns list of quadrature weights for 
        face_index. 
        """
        return self.face_quadrature_weights[face_index]

    def has_face_quadrature(self, face_index):
        """
        Returns True if face_index has associated 
        quadrature points and weights. 
        """
        return self.face_quadrature_points.has_key(face_index)

    def has_cell_quadrature(self, cell_index):
        """ Returns True if cell_index has associated 
        quadrature points and weights. 
        """
        return self.cell_quadrature_points.has_key(cell_index)

    def set_cell_quadrature_points(self, cell_index, points):
        """ Sets quadrature points for cell_index. The points are 
        a list of coordinates (numpy arrays). Quadrature
        is used to compute volume integrals for 
        integrating the forcing function over the volume 
        of the cell. In addition, it can be used to 
        compute the pressure error against known solutions. 
        """
        self.cell_quadrature_points[cell_index] = points

    def get_cell_quadrature_points(self, cell_index):
        """ Returns the quadrature points for cell_index. 
        """
        return self.cell_quadrature_points[cell_index]

    def set_cell_quadrature_weights(self, cell_index, weights):
        """ Sets quadraturs weights for cell_index. The points are 
        a list of coordinates (numpy arrays). Quadrature
        is used to compute volume integrals for 
        integrating the forcing function over the volume 
        of the cell. In addition, it can be used to 
        compute the pressure error against known solutions. 
        """
        self.cell_quadrature_weights[cell_index] = weights


    def get_cell_quadrature_weights(self, cell_index):
        """ Returns the quadrature weights for cell_index. 
        """
        return self.cell_quadrature_weights[cell_index]

    def set_dirichlet_by_face(self, 
                              face_index, 
                              face_orientation, 
                              value):
        """ Directly sets Dirichlet value to face_index. 
        The input *value* corresponding to the 
        integral of the pressure over the face. 
        """        
        self.dirichlet_boundary_values[face_index] = value * \
            self.get_face_area(face_index)*face_orientation

    def set_neumann_by_face(self, 
                              face_index, 
                              face_orientation, 
                              value):
        """ Directly sets Neumann value to face_index. 
        The input *value* corresponding to the 
        integral of the pressure over the face. 
        """        
        self.neumann_boundary_values[face_index] = value 
#        self.dirichlet_boundary_values[face_index] = value * \
#            self.get_face_area(face_index)*face_orientation

        cell_index = self.face_to_cell[face_index][0]
        
        local_face_index = self.get_cell(cell_index).index(face_index)
        
        if self.cell_faces_neumann.has_key(cell_index):
            self.cell_faces_neumann[cell_index].append(face_index)
        else:
            self.cell_faces_neumann[cell_index] = [face_index]


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
        self.dirichlet_boundary_values[face_index] = 0.
        self.dirichlet_boundary_pointers[face_index] = (cell_index, face_orientation)

    def get_dirichlet_pointer_faces(self):
        """ Returns all the faces with Dirichlet 
        values set by pointing to a cell. 
        """
        return self.dirichlet_boundary_pointers.keys()

    def set_face_to_lagrange_pointer(self, face_index, face_orientation, lagrange_index):
        """ Sets face to dirichlet type boundary pointing to 
        lagrange multiplier.
        """        
        # The function adds a zero entry to the 
        # dirichlet_boundary_values dict. This 
        # allows the MFD code to build the matrix 
        # correctly, and doesn't effect the right-hand 
        # side of the problem. 
        self.dirichlet_boundary_values[face_index] = 0.
        self.face_to_lagrange_pointer[face_index] = (lagrange_index, face_orientation)

    def get_all_face_to_lagrange_pointers(self):
        """ Returns all face indices that are 
        pointing to a lagrange multiplier. 
        """
        return self.face_to_lagrange_pointer.keys()
    
    def get_face_to_lagrange_pointer(self, face_index):
        """ Returns the lagrange multiplier index 
        and the face normal orientation. 
        """
        return self.face_to_lagrange_pointer[face_index]
    
    def set_lagrange_to_face_pointers(self, lagrange_index, face_indices, orientations):
        """ Sets the lagrange multiplier to the source faces 
        in order to impose zero flux across the boundary. 
        """
        self.lagrange_to_face_pointers[lagrange_index] = zip(face_indices, orientations)
        
    def get_all_lagrange_to_face_pointers(self):
        """ Returns all lagrange face indices that 
        point to fluxes. 
        """
        return self.lagrange_to_face_pointers.keys()

    def get_lagrange_to_face_pointers(self, lagrange_index):
        """ Returns the faces the lagrange_index face
        points too. 
        """
        return self.lagrange_to_face_pointers[lagrange_index]

    def get_dirichlet_pointer_for_face(self, face_index):
        """ Returns the cell_index for 
        which the Dirichlet boundary will be set 
        implicitly. 
        """
        return self.dirichlet_boundary_pointers[face_index]
    
    def set_neumann_boundary_pointer_to_face(self, face_index, 
                                             face_orientation, 
                                             pointer_index, 
                                             pointer_orientation):
	""" Sets a neumann boundary as a pointer to another
	face in the domain.  
	"""
        self.neumann_boundary_values[face_index] = 0.
        self.neumann_boundary_pointers[face_index] = (pointer_index, -face_orientation*pointer_orientation)

    def get_neumann_pointer_faces(self):
	""" Returns all the faces with Neumann 
        values set by pointing to a cell. 
	"""
        return self.neumann_boundary_pointers.keys()

    def get_neumann_pointer_for_face(self, face_index):
        """ Returns the face_index  for 
        which the Neumann boundary will be set 
        implicitly. 
        """
        return self.neumann_boundary_pointers[face_index]

    def set_forcing_pointer(self, 
                            cell_index, 
                            face_indices,
                            face_orientations):
        """ Sets the value of the forcing function implicity
        as the sum of the fluxes from list of 
        faces. 
        This approach is used for coupling fractures 
        with a reservoir. 
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
        self.cell_forcing_function[cell_index] = 0.
        self.forcing_function_pointers[cell_index] = zip(face_indices, face_orientations)

    def get_forcing_pointer_cells(self):
        """ Returns cell indices with forcing function 
        poitners. 
        """
        return self.forcing_function_pointers.keys()

    def get_forcing_pointers_for_cell(self, cell_index):
        """ Returns list of pointers (face_indices) 
        for cell_index
        """
        return self.forcing_function_pointers[cell_index]
    
    def reset_dirichlet_boundaries(self):
        """
        Resets all Dirichlet bounday data. 
        """
        self.dirichlet_boundary_values = {}
        
    def reset_neumann_boundaries(self):
        """
        Resets all Neumann bounday data. 
        """        
        self.neumann_boundary_values = {}

    def set_cell_domain(self, cell_index, domain):
        """ Sets cell domain identifier 
        for cell_index.
        """
        self.cell_domain[cell_index] = domain

    def get_cell_domain(self, cell_index):
        """ Returns cell domain identifier 
        for cell_index
        """
        return self.cell_domain[cell_index]

    def get_cell_domain_all(self):
        """ Returns list containing 
        all cell_domain tags. 
        """
        return self.cell_domain
    
    def get_cells_in_domain(self, domain):
        """ Returns all cells with domain tag. 
        """
        cells_in_domain = []
        for cell_index in range(self.get_number_of_cells()):
            if self.cell_domain[cell_index]==domain:
                cells_in_domain.append(cell_index)
        return cells_in_domain

    def apply_forcing_from_function(self, forcing_function):
        """ Sets forcing function for entire problem based on 
        a functional representation forcing_function. 
        The forcing_function takes a Numpy array 
        representing a point coordinate and returns 
        the value of the forcing function at that point. 
        The cell_quadrature_points and cell_quadrature_weights 
        are used to approximate the integral 
        of forcing_function over the volume of the cell. 
        """
        for cell_index in range(self.get_number_of_cells()):
            for [quad_point, quad_weight] in \
                    zip(self.get_cell_quadrature_points(cell_index), 
                        self.get_cell_quadrature_weights(cell_index)):
                    self.cell_forcing_function[cell_index] += (forcing_function(quad_point)* 
                                                               quad_weight)

    def apply_forcing_from_grad(self, grad_p, forcing_function):
        """ Computes the source term for a cell from the 
        exact representation of the gradient integrated over the 
        boundaries of the cell. 
        """
        
        for cell_index in range(self.get_number_of_cells()):
            for (face_index, orientation) in \
                    zip(self.get_cell(cell_index), self.get_cell_normal_orientation(cell_index)):
                current_normal = self.get_face_normal(face_index)
                current_center = self.get_face_real_centroid(face_index)
                exact_flux = np.dot(grad_p(current_center), current_normal)*self.get_face_area(face_index)*\
                    orientation
                self.cell_forcing_function[cell_index] += exact_flux

#            for [quad_point, quad_weight] in \
    #                    zip(self.get_cell_quadrature_points(cell_index), 
    #                        self.get_cell_quadrature_weights(cell_index)):
    #                    temp += (forcing_function(quad_point)* 
    #                                                               quad_weight)

    def apply_neumann_from_function(self, boundary_marker, grad_u):
        """ Sets the Neumann boundary values for the
        entire problem based on 
        a functional representation grad_u. All
        boundary faces associated with
        boundary_marker will be set as Neumann 
        The function grad_u takes a Numpy array 
        representing a point coordinate on the 
        face and returns 
        the vector of the Neumann condition at that point. 
        The face_quadrature_points and face_quadrature_weights 
        are used to approximate the 
        normal compontant of grad_u integrated 
        over the face. 
        """
        for [boundary_index, boundary_orientation] in \
                self.get_boundary_faces_by_marker(boundary_marker):
            neumann_value = 0.
            for (quad_points, quad_weights) in \
                    zip(self.get_face_quadrature_points(boundary_index), 
                        self.get_face_quadrature_weights(boundary_index)):
                    neumann_value += \
                        (np.dot(grad_u(quad_points),\
                                    self.get_face_normal(boundary_index))*\
                             quad_weights)

            self.neumann_boundary_values[boundary_index] = neumann_value/self.get_face_area(boundary_index)
            cell_index = self.face_to_cell[boundary_index][0]

            local_face_index = self.get_cell(cell_index).index(boundary_index)

            if self.cell_faces_neumann.has_key(cell_index):
                self.cell_faces_neumann[cell_index].append(boundary_index)
            else:
                self.cell_faces_neumann[cell_index] = [boundary_index]

    def apply_dirichlet_from_function(self, boundary_marker, p_function): 
        """ Sets the Dirichlet boundary values for the
        entire problem based on 
        a functional representation p_function. All
        boundary faces associated with
        boundary_marker will be set as Dirichlet. 
        The p_function takes a Numpy array 
        representing a point coordinate on the 
        face and returns the scalar value of
        the Dirichlet condition at that point. 
        The face_quadrature_points and face_quadrature_weights 
        are used to approximate the 
        the integral of p_function over the face. 
        """
        for (boundary_index, boundary_orientation) \
                in self.get_boundary_faces_by_marker(boundary_marker):

            quad_sum = 0.
            
            for (quad_points, quad_weights) \
                    in zip(self.get_face_quadrature_points(boundary_index), 
                           self.get_face_quadrature_weights(boundary_index)):

                    quad_sum += quad_weights * p_function(quad_points)

            self.dirichlet_boundary_values[boundary_index] = \
                quad_sum*boundary_orientation

    def apply_neumann_from_list(self, boundary_index, grad_u_values):
        """ Not implemented yet. 
        """
        pass


    def get_dirichlet_boundary_values(self):
        """ Returns a list of all faces designated as 
        Dirichlet boundaries and their values. 
        """
        return self.dirichlet_boundary_values.iteritems()

    def get_dirichlet_boundary_value_by_face(self, face_index):
        """ Return pressure value at face_index
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

    def add_rate_well(self, injection_rate, cell_index, well_name):
        """ Adds rate specified well at the center of cell_index.
        Returns the index of the new rate well. 
        """
        pass 

    def add_point_rate_well(self, injection_rate, cell_index, well_name):
        """ Adds rate specified point source at the center of cell_index.
        Returns the index of the new rate well. 
        """
        self.rate_wells_cell_index.append(cell_index)
        self.rate_wells_rate.append(injection_rate)
        self.rate_wells_name.append(well_name)
        return len(self.rate_wells)-1

    def set_gravity_vector(self, gravity_vector):
        """ Set vector indicating gravity acceleration direction. 
        """
        self.gravity_vector = gravity_vector
        
    def get_gravity_vector(self):
        """ Returns gravity vector (down direction)
        """
        return self.gravity_vector

    def set_fluid_density(self, fluid_density):
        """ Set fluid density used for gravity computations. 
        This density will not factor into \frac{K\rho}{\mu}, 
        that still must be set directly by the model or 
        by factoring it into K. 
        """
        self.fluid_density = fluid_density
        
    def get_fluid_density(self):
        """ Returns the fluid density. 
        """
        return self.fluid_density

    def set_gravity_acceleration(self, gravity_acceleration):
        """ Sets the gravity acceleration constant. 
        """
        self.gravity_acceleration = gravity_acceleration

    def get_gravity_acceleration(self):
        """ Returns the gravity acceleration constant. 
        """
        return self.gravity_acceleration

    def set_use_gravity(self,  setting):
        """ Toggles whether gravity is added by MFD. 
        """
        self.use_gravity = setting

    def is_using_gravity(self):
        """ Rerturns whether model is to use
        gravity. 
        """
        return self.use_gravity

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
        raise Exception("Couldn't compute normal for face " + srt(face_index))
       

    def find_face_centroid(self, face_index):
        """ Returns centroid coordinates for face_index. 
        This function assumes planarity of the face. 
        and is currently intended for use with three dimensional 
        meshes. It uses the first two edges of the face as 
        vectors, and the rest of the face is projected onto 
        the 2D surface defined by those two edges. 
        The function returns the area of the face, as well
        as the x, y, z coordinates of its center. 
        """
        if self.dim == 3:
            polygon = map(lambda x: np.array(self.get_point(x)), 
                          self.get_face(face_index))
            
            v1 = polygon[1]-polygon[0]
            v2 = polygon[-1]-polygon[0]

            assert(np.linalg.norm(v2) >1.e-12)
            assert(np.linalg.norm(v1) >1.e-12)
            
            v1 = v1/np.linalg.norm(v1)

            v_temp = np.cross(v1,v2)
            v2 = np.cross(v_temp, v1)


            if np.linalg.norm(v2)< 1.e-10:
                v2 = polygon[-2]-polygon[-1]
                v_temp = np.cross(v1,v2)
                v2 = np.cross(v_temp, v1)
            
            v2 = v2/np.linalg.norm(v2)

            origin = polygon[0]

            transposed_polygon = map(lambda x: x - origin, polygon)
            polygon_projected_v1 = map(lambda x: np.dot(x, v1), 
                                       transposed_polygon)
            polygon_projected_v2 = map(lambda x: np.dot(x, v2), 
                                       transposed_polygon)
            polygon_projected =  zip(polygon_projected_v1,  
                                     polygon_projected_v2)

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
            
            return (area, np.array([centroid_3d_x, centroid_3d_y, centroid_3d_z]))
        
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
        return abs(area)

    def find_volume_centroid(self, cell_index):
        """ Returns the volume and centroid for a 3D cell_index. 
        Based on code and paper by Brian Mirtich. 
        """
        
        volume = 0.
        centroid = np.zeros(3)

        for (face_index, face_orientation) in zip(self.get_cell(cell_index), 
                                                  self.get_cell_normal_orientation(cell_index)):

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
                next_points = points[1:]+points[:1]
            else:
                next_points = self.get_face(face_index)
                points = next_points[1:]+next_points[:1]
            
            for (point_index, next_point_index) in zip(points, next_points):
                a0 = self.get_point(point_index)[A]
                b0 = self.get_point(point_index)[B]
                a1 = self.get_point(next_point_index)[A]
                b1 = self.get_point(next_point_index)[B]
                da = a1-a0;
                db = b1-b0;
                a0_2 = a0*a0 
                a0_3 = a0_2*a0 
                a0_4 = a0_3*a0
                b0_2 = b0*b0 
                b0_3 = b0_2*b0 
                b0_4 = b0_3*b0
                a1_2 = a1*a1
                a1_3 = a1_2*a1 
                b1_2 = b1*b1
                b1_3 = b1_2*b1;
                C1 = a1 + a0;
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

            w = -current_normal.dot(self.get_point(self.get_face(face_index)[0]))
            k1 = 1./current_normal[C]
            k2 = k1*k1
            k3 = k2*k1
            k4 = k3*k1
            
            Fa = k1*Pa
            Fb = k1*Pb
            Fc = -k2*(current_normal[A]*Pa + current_normal[B]*Pb + w*P1)

            Faa = k1*Paa;
            Fbb = k1*Pbb;
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

    def find_volume_centroid_old(self, cell_index):
        """ Returns the volume and centroid for a 3D cell_index. 
        Uses volInt function in C and passes informaiton 
        using file transfers. 
        """
        temp_input = open("temp.dat", 'w')

        current_cell = self.get_cell(cell_index)
        current_cell_orientations = self.get_cell_normal_orientation(cell_index)

        global_to_local_point_mapping = {}
        local_to_global_point_mapping = {}
        current_vertex_count = 0

        for face_index in current_cell:
            for edge in self.get_face(face_index):
                if not global_to_local_point_mapping.has_key(edge):
                    global_to_local_point_mapping[edge] = current_vertex_count
                    local_to_global_point_mapping[current_vertex_count] = edge
                    current_vertex_count += 1

        print >> temp_input , len(global_to_local_point_mapping)
        print >> temp_input 

        for index in range(len(local_to_global_point_mapping)):
            global_index = local_to_global_point_mapping[index]
            print >> temp_input ,  " ".join(map(
                    lambda x: str(x), self.get_point(global_index)))

        print >> temp_input 
        print >> temp_input, len(current_cell)
        print >> temp_input 

        ## Must check that the face is numbered in the right-hand
        ## rule relative to the normal*orientation.
        for (face_index, face_orientation) in \
                zip(current_cell, current_cell_orientations):

            print >> temp_input , len(self.get_face(face_index)), 
            
            if face_orientation > 0:
                for edge in self.get_face(face_index):
                    print >> temp_input, global_to_local_point_mapping[edge], 
                print >> temp_input 

            else:
                for edge in reversed(self.get_face(face_index)):
                    print >> temp_input, global_to_local_point_mapping[edge], 
                print >> temp_input 

        temp_input.close()

        output = os.popen("./volInt temp.dat")

        for i in range(3):
            output.readline()

        volume_line = output.readline().split()
        volume = abs(float(volume_line[-1]))

        for i in range(13):
            output.readline()

        full_list = output.readline().split()

        center_x = float(full_list[4][:-1])
        center_y = float(full_list[5][:-1])
        center_z = float(full_list[6][:-2])

        return (volume, np.array([center_x, center_y, center_z]))

    def add_scalar_to_matplotlib(self, cell_data):
        """ Method for visualizing scalar data in 
        general 2D meshes using matplotlib. 
        """
        fig=pylab.figure()
        ax=fig.add_subplot(111)
        patches = map(lambda x:Polygon(x, True), self.two_d_polygons)
        p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=1.0)
        p.set_array(cell_data)
        ax.add_collection(p)
        p.set_clim([min(cell_data), max(cell_data)])
        pylab.colorbar(p)

    def add_vector_to_matplotlib(self, face_data):
        """ Method for visualizing vector data in 
        general 2D meshes using matplotlib. 
        """
        max_velocity = max(map(abs, face_data))
        min_edge_length = min(self.face_areas)
        vector_scale = max_velocity/min_edge_length*.5
        print "maxVelocity", max_velocity
        vectors = map(lambda x: x[0]*x[1], zip(face_data, self.face_normals))
        pylab.quiver(map(lambda x: x[0], self.face_real_centroids), 
                     map(lambda x: x[1], self.face_real_centroids),
                     map(lambda x: x[0], vectors),
                     map(lambda x: x[1], vectors),
                     angles='xy',scale = vector_scale,color='r')

    def show_matplotlib(self, file_name = None):
        """ Displays the 2D mesh data that was set using 
        add_scalar_to_matplotlib and add_vector_to_matplotlib. 
        If no file_name is specified, the plot will show 
        as a pop up during runtime. 
        """
        if file_name is not None:
            pylab.savefig(file_name)
        pylab.show()

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
        output = open(file_name +".vtk",'w')

        print >> output, "# vtk DataFile Version 1.0"
        print >> output, "MFD output"
        print >> output, "ASCII"
        print >> output, "DATASET UNSTRUCTURED_GRID"
        print >> output, "POINTS", self.get_number_of_faces() ,  "float"
        
        for face_index in range(self.get_number_of_faces()):
            current_point = self.get_face_real_centroid(face_index)
            print >> output, current_point[0], 
            print >> output, current_point[1],
            print >> output, current_point[2]

        print >> output, " "
        print >> output, "CELLS", self.get_number_of_faces(),
        print >> output,  self.get_number_of_faces()*2

        for face_index in range(self.get_number_of_faces()):
            print >> output, "1", face_index + 1
        print >> output, " "
        print >> output, "CELL_TYPES" , self.get_number_of_faces()

        for index in range(self.get_number_of_faces()):
            print >> output, "1"
        print >> output, " " 
        print >> output, "POINT_DATA", number_of_faces
        print >> output, " " 

        for data_index in range(len(vector_labels)):
            print >> output, "VECTORS", vector_labels[data_index], "float"
            
            for face_index in range(len(vector_magnitudes[data_index])):
                current_vector = vector_magnitudes[data_index][face_index] * \
                    self.get_face_normal(face_index)

                print >> output, current_vector[0], 
                print >> output, current_vector[1],
                print >> output, current_vector[2]

            print >>output, " " 
    
    def output_cell_normals(self, file_name, cell_index):
        """ Outputs the normals over the cell in the outward direction. 
        The function is intended for checking the correct orienation of cell. 
        """
        output = open(file_name +".vtk",'w')

        number_of_faces = len(self.get_cell(cell_index))

        print >> output, "# vtk DataFile Version 1.0"
        print >> output, "MFD output"
        print >> output, "ASCII"
        print >> output, "DATASET UNSTRUCTURED_GRID"
        print >> output, "POINTS", number_of_faces ,  "float"
        
        for face_index in self.get_cell(cell_index):
            centroid = self.get_face_real_centroid(face_index)
            print >> output, centroid[0], 
            print >> output, centroid[1], 
            print >> output, centroid[2]

        print >> output, " "
        print >> output, "CELLS", number_of_faces,
        print >> output, number_of_faces*2

        for index in range(number_of_faces):
            print >> output, "1", index+1
        print >> output, " " 
        print >> output, "POINT_DATA", number_of_faces
        print >> output, " " 
            
        print >> output, "VECTORS", "OUT_NORMAL", "float"

        for (face_index, orientation) in zip(self.get_cell(cell_index), 
                                         self.get_cell_normal_orientation(cell_index)):
            normal = self.get_face_normal(face_index)
    
            print >> output, normal[0]*orientation, 
            print >> output, normal[1]*orientation, 
            print >> output, normal[2]*orientation

        print >>output, " " 

 
    def output_vtk_faces(self, file_name, face_indices, face_values = [], face_value_labels = []):
        """ Outputs in vtk format the faces in face_indices. 
        """
        output = open(file_name +".vtk",'w')
        print >> output, "# vtk DataFile Version 2.0"
        print >> output, "# unstructured mesh"
        print >> output, "ASCII"
        print >> output, "DATASET UNSTRUCTURED_GRID"
        print >> output, "POINTS", self.get_number_of_points(), "float"

        for point_index in range(self.get_number_of_points()):
            point = self.get_point(point_index)
            print >> output, point[0], point[1], point[2]

        total_polygon_points = 0
        for face_index in face_indices:
            total_polygon_points += \
                self.get_number_of_face_points(face_index)+1

        print >> output, "CELLS", len(face_indices)
        print >> output, total_polygon_points
        
        for face_index in face_indices:
            current_face = self.get_face(face_index)
            print >> output, len(current_face), 
            for point in current_face:
                print >> output, point, 
            print >> output, "\n", 

        print >> output, "CELL_TYPES", len(face_indices)
        for face_index in face_indices:
            print >> output, 7
            
        if face_values:
            print >>output, "CELL_DATA", len(face_indices)
            for (entry, entryname) in zip(face_values, face_value_labels):
                print >>output, "SCALARS", entryname, "double 1"
                print >>output, "LOOKUP_TABLE default" 
                for value in entry:
                    print >>output, value


        output.close()

    def output_vtk_mesh_polygon(self, file_name, cell_values=[], cell_value_labels=[]):
        """ Base implementation for producing 
        vtk files for general polyhedral meshes. 
        The functions draws the unstructured mesh by 
        duplicating the polygonal faces and assigning 
        the values to the faces. 
        """
        output = open(file_name +".vtk",'w')

        print >> output, "# vtk DataFile Version 2.0"
        print >> output, "# unstructured mesh"
        print >> output, "ASCII"
        print >> output, "DATASET UNSTRUCTURED_GRID"
        print >> output, "POINTS", self.get_number_of_points(), "float"

        for point_index in range(self.get_number_of_points()):
            point = self.get_point(point_index)
            print >> output, point[0], point[1], point[2]

        total_polygon_points = 0
        for cell_index in range(self.get_number_of_cells()):
            for face_index in self.get_cell(cell_index):
                total_polygon_points += \
                    self.get_number_of_face_points(face_index)+1

        print >> output, "CELLS", self.get_number_of_faces()*2-self.get_number_of_boundary_faces(), 
        print >> output, total_polygon_points
        
        for cell_index in range(self.get_number_of_cells()):
            for face_index in self.get_cell(cell_index):
                current_face = self.get_face(face_index)
                print >> output, len(current_face), 
                for point in current_face:
                    print >> output, point, 
                print >> output, "\n", 

        print >> output, "CELL_TYPES", self.get_number_of_faces()*2-self.get_number_of_boundary_faces()
        for face_index in range(self.get_number_of_faces()*2-self.get_number_of_boundary_faces()):
            print >> output, 7

        if cell_values:
            print >>output, "CELL_DATA", self.get_number_of_faces()*2-self.get_number_of_boundary_faces()
            for (entry, entryname) in zip(cell_values, cell_value_labels):
                print >>output, "SCALARS", entryname, "double 1"
                print >>output, "LOOKUP_TABLE default" 
                for (cell_index, value) in enumerate(entry):
                    for face_index in self.get_cell(cell_index):
                        print >>output, value
                        
        output.close()

    def output_vtk_mesh(self, file_name, cell_values=[], cell_value_labels=[]):
        """ Base implementation for producing 
        vtk files for general polyhedral meshes. 
        """
        print "Parent VTK"
        output = open(file_name +".vtk",'w')
        print >> output, "# vtk DataFile Version 2.0"
        print >> output, "# unstructured mesh"
        print >> output, "ASCII"
        print >> output, "DATASET UNSTRUCTURED_GRID"
        print >> output, "POINTS", self.get_number_of_points(), "float"

        for point_index in range(self.get_number_of_points()):
            point = self.get_point(point_index)
            print >> output, point[0], point[1], point[2]

        total_polygon_points = 0
        for cell_index in range(self.get_number_of_cells()):
            for face_index in self.get_cell(cell_index):
                total_polygon_points+=\
                    self.get_number_of_face_points(face_index)+1
            total_polygon_points+=2
            
        print >> output, "CELLS", self.get_number_of_cells(), 
        print >> output, total_polygon_points
        
        for cell_index in range(self.get_number_of_cells()):
            number_of_entries = len(self.get_cell(cell_index))+\
                sum([self.get_number_of_face_points(face_index) for face_index in self.get_cell(cell_index)])+1
            print >>output, number_of_entries,
            print >>output, len(self.get_cell(cell_index)), 
            for face_index in self.get_cell(cell_index):
                current_face = self.get_face(face_index)
                print >> output, len(current_face), 
                for point in current_face:
                    print >> output, point, 
            print >> output, "\n", 
            
        print >> output, "CELL_TYPES", self.get_number_of_cells()
        for cell_index in range(self.get_number_of_cells()):
            print >> output, 42

        if cell_values:
            print >>output, "CELL_DATA", self.get_number_of_cells()
            for (entry, entryname) in zip(cell_values, cell_value_labels):
                print >>output, "SCALARS", entryname, "double 1"
                print >>output, "LOOKUP_TABLE default" 
                for value in entry:
                    print >>output, value
                    
        output.close()

    def output_scalar_gnuplot(self, cell_values , filename):
        """ Outputs scalar values associated with mesh to 
        gnuplot. Only works for 2D meshes. 
        """
        if self.mesh.dim == 2:
            gnuout  = open(filename+'.dat','w')
            for cell_index in range(self.mesh.get_number_of_cells()):
                cellCenter =  self.mesh.get_cell_real_centroid(cell_index)
                print >>gnuout, cellCenter[0], cellCenter[1], 
                print >>gnuout, cel_values[cell_index]
        else:
            print "no 3D gnuplot output"

    def find_cell_near_point(self, point):
        """
        Returns cell whose centroid is closest 
        to a given point. 
        """
        closest_cell = 0
        min_distance = np.linalg.norm(self.get_cell_real_centroid(0)-point)
        for cell_index in range(1, self.get_number_of_cells()):
            new_distance = np.linalg.norm(self.get_cell_real_centroid(cell_index)-point)
            if new_distance < min_distance:
                closest_cell = cell_index
                min_distance = new_distance

        return closest_cell
        

    def nonplanar_normal(self, face):
        for i in range(1):
            v1 = self.get_point(face[i+1]) - self.get_point(face[i]) 
            v2 = self.get_point(face[i]) - self.get_point(face[i-1]) 
            normal = np.cross(v2, v1)
            
        return normal/np.linalg.norm(normal)

    def nonplanar_face_centroid(self, face):
        p1 = self.get_point(face[0])
        p2 = self.get_point(face[1])
        p3 = self.get_point(face[2])
        p4 = self.get_point(face[3])
        
        centerPoint = .25 * (p1 + p2 + p3 + p4)
        
        return centerPoint
    
    def nonplanar_cell_centroid(self, cell):
        centroid = np.zeros(3)
        count = 0.
        for face in cell:
            for point in self.get_face(face):
                count += 1.
                centroid += self.get_point(point)
                
        centroid = centroid/count
        return centroid

    def nonplanar_area(self, face):
        area = 0.

        p1 = self.points[face[0]]
        p2 = self.points[face[1]]
        p3 = self.points[face[2]]
        p4 = self.points[face[3]]
        
        centerPoint = .25 * (p1 + p2 + p3 + p4)
        
        a = np.linalg.norm(p1-p2)
        b = np.linalg.norm(p2-centerPoint)
        c = np.linalg.norm(centerPoint - p1)
        s = (a + b + c)/2.

        area += math.sqrt(s * (s-a) * (s - b) * (s - c))

        a = np.linalg.norm(p2-p3)
        b = np.linalg.norm(p3-centerPoint)
        c = np.linalg.norm(centerPoint - p2)
        s = (a + b + c)/2.

        area += math.sqrt(s * (s-a) * (s - b) * (s - c))

        a = np.linalg.norm(p3-p4)
        b = np.linalg.norm(p4-centerPoint)
        c = np.linalg.norm(centerPoint - p3)
        s = (a + b + c)/2.

        area += math.sqrt(s * (s-a) * (s - b) * (s - c))

        a = np.linalg.norm(p4-p1)
        b = np.linalg.norm(p1-centerPoint)
        c = np.linalg.norm(centerPoint - p4)
        s = (a + b + c)/2.

        area += math.sqrt(s * (s-a) * (s - b) * (s - c))

        return area 

    def subdivide_by_domain(self, cells, domain_number):
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
                    print cell1, cell2
                    if cell1 == cell_index:
                        if cell2 not in cells:
                            lagrange_faces.append(face_index)
                    if cell2 == cell_index:
                        if cell1 not in cells:
                            lagrange_faces.append(face_index)

        
        print "lagrange_faces"
        print lagrange_faces
        
        for face_index in lagrange_faces:
            (cell1, cell2) = self.face_to_cell[face_index]
            if cell1 in cells:
                this_cell = cell1
                other_cell = cell2
            else:
                this_cell = cell2
                other_cell = cell1
                
            new_face_index = self.add_face(self.get_face(face_index))
            lagrange_face_index = self.add_face(self.get_face(face_index))
            
            self.set_face_normal(new_face_index, self.get_face_normal(face_index))
            self.set_face_normal(lagrange_face_index, self.get_face_normal(face_index))
            
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
            
            self.set_face_quadrature_points(new_face_index, 
                                            [self.get_face_real_centroid(new_face_index)])
            print "error in quadrature weights "
            self.set_face_quadrature_weights(new_face_index, 
                                             [self.get_face_real_centroid(new_face_index)])
            
            local_face_index_in_other = self.get_cell(other_cell).index(face_index)
            new_cell_faces = self.get_cell(other_cell)
            new_cell_faces[local_face_index_in_other] = new_face_index
            

    def build_frac_from_faces(self, faces, boundary_locations = []):
        """ Takes a list of face indices, and 
        extrudes them into cells. 
        """
        boundary_markers = self.get_boundary_markers()
        ## All fracture boundaries share the same marker. 
        assert(len(boundary_markers)+1 not in boundary_markers)
        fracture_boundary_marker = len(boundary_markers)+1
        self.add_boundary_marker(fracture_boundary_marker, "fracture_boundaries")

        connections = []
        non_connected_edges = []        
        for face in faces:
            non_connected_edges.append([])
            for local_edge_index in range(len(self.get_face(face))):
                non_connected_edges[-1].append(local_edge_index)

        print  "faces", faces
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

        print connections
                    
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

        multiple_connection_groups = filter(lambda x:len(x)>0, multiple_connection_groups)

        new_multiple_connection_groups = []
        ## Switch from connection index to actual connections
        for group in multiple_connection_groups:
            new_multiple_connection_groups.append([])
            for connection_index in group:
                new_multiple_connection_groups[-1].append(list(connections[connection_index]))
                
        multiple_connection_groups = new_multiple_connection_groups
        print "multiple_connection_groups", multiple_connection_groups
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

                print done_faces
                if ((current_face, 'TOP')) not in done_faces:
                    subface_connections.append(group[max_top_connection_index]+['TOP'])
                    done_faces.append((current_face, 'TOP')) 
                    if group[max_top_connection_index][6]==0:
                        done_faces.append((face2_top, 'TOP')) 
                    else:
                        done_faces.append((face2_top, 'BOT')) 

                if ((current_face, 'BOT')) not in done_faces:
                    subface_connections.append(group[max_bot_connection_index]+['BOT'])        
                    done_faces.append((current_face, 'BOT'))
                    if group[max_bot_connection_index][6]==0:
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
        
        print "connections line 1966", connections
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
        temp_out = open("tempfaces", 'w')
        for face in range(len(faces)):
            face_to_walls[face] = []
            top_points.append([-1]*len(self.get_face(faces[face])))
            bot_points.append([-1]*len(self.get_face(faces[face])))
            mid_points.append([-1]*len(self.get_face(faces[face])))
                
        for connection in connections:
            norm1 = np.zeros(3)
            for (local_face_index, orientation) in connection[7]:
                if orientation == 0: 
                    norm1+= self.get_face_normal(faces[local_face_index])
                else:
                    norm1-= self.get_face_normal(faces[local_face_index])

            norm2 = np.zeros(3)
            for (local_face_index, orientation) in connection[8]:
                if orientation == 0: 
                    norm2+= self.get_face_normal(faces[local_face_index])
                else:
                    norm2-= self.get_face_normal(faces[local_face_index])
            
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
            
            new_face_index =self.add_face([point_1_index, 
                                           point_2_index, 
                                           point_3_index, 
                                           point_4_index,])
            
            (area, centroid) =  self.find_face_centroid(new_face_index)
            self.set_face_normal(new_face_index, 
                                 self.nonplanar_normal(self.get_face(new_face_index)))

            self.set_face_real_centroid(new_face_index, centroid)
                                        #self.nonplanar_face_centroid(self.get_face(new_face_index)))
            if self.has_face_shifted_centroid:
                self.set_face_shifted_centroid(new_face_index, centroid)
                                            #self.nonplanar_face_centroid(self.get_face(new_face_index)))

            self.set_face_area(new_face_index, area)
                               #self.nonplanar_area(self.get_face(new_face_index)))

            face_to_walls[connection[0]].append((new_face_index, 1))
            face_to_walls[connection[1]].append((new_face_index, -1))

            top_points[connection[0]][connection[2]]=point_4_index
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
        print "subface_connections"
        print subface_connections
        for connection in subface_connections:
            norm1 = np.zeros(3)
            for (local_face_index, orientation) in connection[8]:
                if orientation == 0: 
                    norm1+= self.get_face_normal(faces[local_face_index])
                else:
                    norm1-= self.get_face_normal(faces[local_face_index])

            norm2 = np.zeros(3)
            for (local_face_index, orientation) in connection[8]:
                if orientation == 0: 
                    norm2+= self.get_face_normal(faces[local_face_index])
                else:
                    norm2-= self.get_face_normal(faces[local_face_index])
            
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
                
            new_face_index =self.add_face([point_1_index, 
                                           point_2_index, 
                                           point_3_index, 
                                           point_4_index,])

            (area, centroid) =  self.find_face_centroid(new_face_index)
            self.set_face_normal(new_face_index, 
                                 self.nonplanar_normal(self.get_face(new_face_index)))
            self.set_face_real_centroid(new_face_index, centroid)
                                        #self.nonplanar_face_centroid(self.get_face(new_face_index)))
            
            if self.has_face_shifted_centroid:
                self.set_face_shifted_centroid(new_face_index, centroid)
                                            #self.nonplanar_face_centroid(self.get_face(new_face_index)))
            
                
            self.set_face_area(new_face_index, area)
                               #self.nonplanar_area(self.get_face(new_face_index)))

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

#        self.output_vtk_faces("new_faces", new_faces)                
#        1/0

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

                #new_face_points = [point_1_index, 
                #                   point_2_index, 
                #                   point_3_index, 
                #                   point_4_index,]

                new_face_index =self.add_face(new_face_points)
                normal = self.nonplanar_normal(new_face_points)
                self.set_face_normal(new_face_index, normal)
                                     
                (area, centroid) =  self.find_face_centroid(new_face_index)
                centroid = self.nonplanar_face_centroid(new_face_points)
                self.set_face_real_centroid(new_face_index, centroid)
                                            
                if self.has_face_shifted_centroid:
                    self.set_face_shifted_centroid(new_face_index, centroid)
                                                   #self.nonplanar_face_centroid(new_face_points))
                
                self.set_face_area(new_face_index, area)
                                   #self.nonplanar_area(new_face_points))
                
                current_fracture_boundary = fracture_boundary_marker
                for b_case in boundary_locations:
                    condition = b_case[0]
                    if condition(centroid):
                        print "ADDING FRAC BOUNDARY", 
                        current_fracture_boundary = b_case[1]
                
                if np.dot(normal, centroid -self.get_face_real_centroid(global_face_index))>  0.:
                    self.add_boundary_face(current_fracture_boundary, new_face_index, 1)
                    print "ONE"
                else:
                    self.add_boundary_face(current_fracture_boundary, new_face_index, -1)
                    print "TWO"

                self.set_face_quadrature_points(new_face_index, 
                                                [self.get_face_real_centroid(new_face_index)])
                self.set_face_quadrature_weights(new_face_index, 
                                                 [self.get_face_area(new_face_index)])

                face_to_walls[local_face_index].append((new_face_index, 1))

                new_faces.append(new_face_index)


        ## Adds the top and bottom faces. 
        for local_face_index in range(len(faces)):
            new_face_points = top_points[local_face_index]
            new_face_index = self.add_face(new_face_points)
            face_to_walls[local_face_index].append((new_face_index, 1))
            self.add_boundary_face(fracture_boundary_marker, new_face_index, 1)

            (area, centroid) =  self.find_face_centroid(new_face_index)
            self.set_face_normal(new_face_index, 
                                 self.nonplanar_normal(new_face_points))
            self.set_face_real_centroid(new_face_index, centroid)
                                        #self.nonplanar_face_centroid(new_face_points))
            if self.has_face_shifted_centroid:
                self.set_face_shifted_centroid(new_face_index, centroid)
                                               #self.nonplanar_face_centroid(new_face_points))
                
            self.set_face_area(new_face_index, area)
                               #self.nonplanar_area(new_face_points))

            self.set_face_quadrature_points(new_face_index, 
                                            [self.get_face_real_centroid(new_face_index)])
            self.set_face_quadrature_weights(new_face_index, 
                                             [self.get_face_area(new_face_index)])

            new_faces.append(new_face_index)

            new_face_points = bot_points[local_face_index]
            new_face_index = self.add_face(new_face_points)
            face_to_walls[local_face_index].append((new_face_index, -1))

            self.add_boundary_face(fracture_boundary_marker, new_face_index, 1)

            (area, centroid) =  self.find_face_centroid(new_face_index)

            self.set_face_normal(new_face_index, 
                                 self.nonplanar_normal(new_face_points))

            self.set_face_real_centroid(new_face_index, centroid)
                                        #self.nonplanar_face_centroid(new_face_points))
            if self.has_face_shifted_centroid:
                self.set_face_shifted_centroid(new_face_index, centroid)
                                               #self.nonplanar_face_centroid(new_face_points))

            self.set_face_area(new_face_index, area)
                               #self.nonplanar_area(new_face_points))
            self.set_face_quadrature_points(new_face_index, [centroid])
                                            #[self.get_face_real_centroid(new_face_index)])
            self.set_face_quadrature_weights(new_face_index, [area])
                                             #[self.get_face_area(new_face_index)])

            new_faces.append(new_face_index)
        
        # Duplicate reservoir face for interior dirichlet boundary. 
        for face in face_to_walls:
            top_res_face_index = faces[face]

            if len(self.face_to_cell[top_res_face_index]) == 2:
                bot_res_face_index = self.add_face(list(self.get_face(top_res_face_index)))
                self.set_face_area(bot_res_face_index, self.get_face_area(top_res_face_index))
                self.set_face_normal(bot_res_face_index, self.get_face_normal(top_res_face_index))
                self.set_face_real_centroid(bot_res_face_index, 
                                            self.get_face_real_centroid(top_res_face_index))
                if self.has_face_shifted_centroid:
                    self.set_face_shifted_centroid(bot_res_face_index, 
                                                   self.get_face_real_centroid(top_res_face_index))

                bottom_cell = self.face_to_cell[top_res_face_index][1]

                new_cell_faces = array.array('i', self.get_cell(bottom_cell))
                local_face_index_in_cell = new_cell_faces.index(top_res_face_index)

                new_cell_faces[local_face_index_in_cell] = bot_res_face_index

                top_cell_index = self.face_to_cell[top_res_face_index][0]
                local_top_face_index_in_cell = self.get_cell(top_cell_index).index(top_res_face_index)
                top_res_face_orientation =\
                    self.get_cell_normal_orientation(top_cell_index)[local_top_face_index_in_cell]
                                
                self.face_to_cell[top_res_face_index].remove(bottom_cell)
                self.set_cell_faces(bottom_cell, new_cell_faces)
            else:
                print "single cell face..."
                1/0
                
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

        self.apply_neumann_from_function(fracture_boundary_marker, lambda x:np.zeros(3))
        self.output_vtk_faces("new_faces", new_faces)
            



    def build_frac_from_faces_single(self, faces, boundary_locations = []):
        """ Takes a list of face indices, and 
        extrudes them into cells. In this version of the function, 
        a single fracture cell is created. 
        """
        boundary_markers = self.get_boundary_markers()
        ## All fracture boundaries share the same marker. 
        assert(len(boundary_markers)+1 not in boundary_markers)
        fracture_boundary_marker = len(boundary_markers)+1
        self.add_boundary_marker(fracture_boundary_marker, "fracture_boundaries")

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

        multiple_connection_groups = filter(lambda x:len(x)>0, multiple_connection_groups)

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
                    if group[max_top_connection_index][6]==0:
                        done_faces.append((face2_top, 'TOP')) 
                    else:
                        done_faces.append((face2_top, 'BOT')) 

                if ((current_face, 'BOT')) not in done_faces:
                    subface_connections.append(group[max_bot_connection_index]+['BOT'])        
                    done_faces.append((current_face, 'BOT'))
                    if group[max_bot_connection_index][6]==0:
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
        temp_out = open("tempfaces", 'w')
        for face in range(len(faces)):
            face_to_walls[face] = []
            top_points.append([-1]*len(self.get_face(faces[face])))
            bot_points.append([-1]*len(self.get_face(faces[face])))
            mid_points.append([-1]*len(self.get_face(faces[face])))
                
        for connection in connections:
            norm1 = np.zeros(3)
            for (local_face_index, orientation) in connection[7]:
                if orientation == 0: 
                    norm1+= self.get_face_normal(faces[local_face_index])
                else:
                    norm1-= self.get_face_normal(faces[local_face_index])

            norm2 = np.zeros(3)
            for (local_face_index, orientation) in connection[8]:
                if orientation == 0: 
                    norm2+= self.get_face_normal(faces[local_face_index])
                else:
                    norm2-= self.get_face_normal(faces[local_face_index])
            
            norm1 /= np.linalg.norm(norm1)
            norm2 /= np.linalg.norm(norm2)

            width = .1
            full_face = self.get_face(faces[connection[0]])
            point1 = -width*norm1+self.get_point(full_face[connection[2]])
            point2 = -width*norm2+self.get_point(full_face[connection[3]%len(full_face)])
            point3 = width*norm2+self.get_point(full_face[connection[3]%len(full_face)])
            point4 = width*norm1+self.get_point(full_face[connection[2]])
            
            point_1_index = self.add_point(point1)
            point_2_index = self.add_point(point2)
            point_3_index = self.add_point(point3)
            point_4_index = self.add_point(point4)
            """
            new_face_index =self.add_face([point_1_index, 
                                           point_2_index, 
                                           point_3_index, 
                                           point_4_index,])
            
            (area, centroid) =  self.find_face_centroid(new_face_index)
            self.set_face_normal(new_face_index, 
                                 self.nonplanar_normal(self.get_face(new_face_index)))

            self.set_face_real_centroid(new_face_index, centroid)
                                        #self.nonplanar_face_centroid(self.get_face(new_face_index)))
            if self.has_face_shifted_centroid:
                self.set_face_shifted_centroid(new_face_index, centroid)
                                            #self.nonplanar_face_centroid(self.get_face(new_face_index)))

            self.set_face_area(new_face_index, area)
                               #self.nonplanar_area(self.get_face(new_face_index)))

            face_to_walls[connection[0]].append((new_face_index, 1))
            face_to_walls[connection[1]].append((new_face_index, -1))
            """

            top_points[connection[0]][connection[2]]=point_4_index
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

            #new_faces.append(new_face_index)

                
        ## Build the subfaces. 
        for connection in subface_connections:
            norm1 = np.zeros(3)
            for (local_face_index, orientation) in connection[8]:
                if orientation == 0: 
                    norm1+= self.get_face_normal(faces[local_face_index])
                else:
                    norm1-= self.get_face_normal(faces[local_face_index])

            norm2 = np.zeros(3)
            for (local_face_index, orientation) in connection[8]:
                if orientation == 0: 
                    norm2+= self.get_face_normal(faces[local_face_index])
                else:
                    norm2-= self.get_face_normal(faces[local_face_index])
            
            norm1 /= np.linalg.norm(norm1)
            norm2 /= np.linalg.norm(norm2)

            width = .1

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
                
            new_face_index =self.add_face([point_1_index, 
                                           point_2_index, 
                                           point_3_index, 
                                           point_4_index,])

            (area, centroid) =  self.find_face_centroid(new_face_index)
            self.set_face_normal(new_face_index, 
                                 self.nonplanar_normal(self.get_face(new_face_index)))
            self.set_face_real_centroid(new_face_index, centroid)
                                        #self.nonplanar_face_centroid(self.get_face(new_face_index)))
            
            if self.has_face_shifted_centroid:
                self.set_face_shifted_centroid(new_face_index, centroid)
                                            #self.nonplanar_face_centroid(self.get_face(new_face_index)))
            
                
            self.set_face_area(new_face_index, area)
                               #self.nonplanar_area(self.get_face(new_face_index)))

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

#        self.output_vtk_faces("new_faces", new_faces)                
#        1/0
            
        ## Adds the bundary faces. 
        for (local_face_index, face) in enumerate(non_connected_edges):
            for point1 in face:
                global_face_index = faces[local_face_index]
                point2 = (point1+1)%len(self.get_face(global_face_index))
                
                new_face_points = []
                norm = self.get_face_normal(global_face_index)
                width = .1
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

                #new_face_points = [point_1_index,
                #                   point_2_index,
                #                   point_3_index,
                #                   point_4_index,]

                new_face_index =self.add_face(new_face_points)
                normal = self.nonplanar_normal(new_face_points)
                self.set_face_normal(new_face_index, normal)
                                     
                (area, centroid) =  self.find_face_centroid(new_face_index)
                centroid = self.nonplanar_face_centroid(new_face_points)
                self.set_face_real_centroid(new_face_index, centroid)
                                            
                if self.has_face_shifted_centroid:
                    self.set_face_shifted_centroid(new_face_index, centroid)
                                                   #self.nonplanar_face_centroid(new_face_points))
                
                self.set_face_area(new_face_index, area)
                                   #self.nonplanar_area(new_face_points))
                
                current_fracture_boundary = fracture_boundary_marker
                for b_case in boundary_locations:
                    condition = b_case[0]
                    if condition(centroid):
                        print "ADDING FRAC BOUNDARY", 
                        current_fracture_boundary = b_case[1]
                
                if np.dot(normal, centroid -self.get_face_real_centroid(global_face_index))>  0.:
                    self.add_boundary_face(current_fracture_boundary, new_face_index, 1)
                else:
                    self.add_boundary_face(current_fracture_boundary, new_face_index, -1)

                self.set_face_quadrature_points(new_face_index, 
                                                [self.get_face_real_centroid(new_face_index)])
                self.set_face_quadrature_weights(new_face_index, 
                                                 [self.get_face_area(new_face_index)])

                face_to_walls[local_face_index].append((new_face_index, 1))

                new_faces.append(new_face_index)


        ## Adds the top and bottom faces. 
        for local_face_index in range(len(faces)):
            new_face_points = top_points[local_face_index]
            new_face_index = self.add_face(new_face_points)
            face_to_walls[local_face_index].append((new_face_index, 1))
            self.add_boundary_face(fracture_boundary_marker, new_face_index, 1)

            (area, centroid) =  self.find_face_centroid(new_face_index)
            self.set_face_normal(new_face_index, 
                                 self.nonplanar_normal(new_face_points))
            self.set_face_real_centroid(new_face_index, centroid)
                                        #self.nonplanar_face_centroid(new_face_points))
            if self.has_face_shifted_centroid:
                self.set_face_shifted_centroid(new_face_index, centroid)
                                               #self.nonplanar_face_centroid(new_face_points))
                
            self.set_face_area(new_face_index, area)
                               #self.nonplanar_area(new_face_points))

            self.set_face_quadrature_points(new_face_index, 
                                            [self.get_face_real_centroid(new_face_index)])
            self.set_face_quadrature_weights(new_face_index, 
                                             [self.get_face_area(new_face_index)])

            #new_faces.append(new_face_index)

            new_face_points = bot_points[local_face_index]
            new_face_index = self.add_face(new_face_points)
            face_to_walls[local_face_index].append((new_face_index, -1))

            self.add_boundary_face(fracture_boundary_marker, new_face_index, 1)

            (area, centroid) =  self.find_face_centroid(new_face_index)

            self.set_face_normal(new_face_index, 
                                 self.nonplanar_normal(new_face_points))

            self.set_face_real_centroid(new_face_index, centroid)
                                        #self.nonplanar_face_centroid(new_face_points))
            if self.has_face_shifted_centroid:
                self.set_face_shifted_centroid(new_face_index, centroid)
                                               #self.nonplanar_face_centroid(new_face_points))

            self.set_face_area(new_face_index, area)
                               #self.nonplanar_area(new_face_points))
            self.set_face_quadrature_points(new_face_index, [centroid])
                                            #[self.get_face_real_centroid(new_face_index)])
            self.set_face_quadrature_weights(new_face_index, [area])
                                             #[self.get_face_area(new_face_index)])

            #new_faces.append(new_face_index)
        
        # Duplicate reservoir face for interior dirichlet boundary. 
        faces_for_cell = []
        normal_orientation_for_cell = []
        new_cell_index = self.add_cell(array.array('i', []), 
                                       array.array('i', []))

        forcing_pointer_faces = []
        forcing_pointer_orientations = []
        for face in face_to_walls:
            top_res_face_index = faces[face]

            if len(self.face_to_cell[top_res_face_index]) == 2:
                bot_res_face_index = self.add_face(list(self.get_face(top_res_face_index)))
                self.set_face_area(bot_res_face_index, self.get_face_area(top_res_face_index))
                self.set_face_normal(bot_res_face_index, self.get_face_normal(top_res_face_index))
                self.set_face_real_centroid(bot_res_face_index, 
                                            self.get_face_real_centroid(top_res_face_index))
                if self.has_face_shifted_centroid:
                    self.set_face_shifted_centroid(bot_res_face_index, 
                                                   self.get_face_real_centroid(top_res_face_index))

                bottom_cell = self.face_to_cell[top_res_face_index][1]

                new_cell_faces = array.array('i', self.get_cell(bottom_cell))
                local_face_index_in_cell = new_cell_faces.index(top_res_face_index)

                new_cell_faces[local_face_index_in_cell] = bot_res_face_index

                top_cell_index = self.face_to_cell[top_res_face_index][0]
                local_top_face_index_in_cell = self.get_cell(top_cell_index).index(top_res_face_index)
                top_res_face_orientation =\
                    self.get_cell_normal_orientation(top_cell_index)[local_top_face_index_in_cell]
                                
                self.face_to_cell[top_res_face_index].remove(bottom_cell)
                self.set_cell_faces(bottom_cell, new_cell_faces)
            else:
                print "single cell face..."
                1/0
                
                
            forcing_pointer_faces += [top_res_face_index, bot_res_face_index] 
            forcing_pointer_orientations += [top_res_face_orientation, -top_res_face_orientation]

            self.set_dirichlet_face_pointer(top_res_face_index, 
                                            top_res_face_orientation, 
                                            new_cell_index) 
            self.set_dirichlet_face_pointer(bot_res_face_index, 
                                            -top_res_face_orientation, 
                                            new_cell_index)
            
            faces_for_cell += [x[0] for x in face_to_walls[face]]
            normal_orientation_for_cell += [x[1] for x in face_to_walls[face]]

        self.output_vtk_faces("cell_faces", faces_for_cell)
        
        self.set_cell_faces(new_cell_index, array.array('i', faces_for_cell))
        self.set_cell_orientation(new_cell_index, array.array('i', normal_orientation_for_cell))

        self.set_forcing_pointer(new_cell_index, 
                                 forcing_pointer_faces, 
                                 forcing_pointer_orientations)

        
        self.set_cell_domain(new_cell_index, 1)
        
        (volume, centroid) = self.find_volume_centroid(new_cell_index)
        self.set_cell_volume(new_cell_index, volume)
        self.set_cell_real_centroid(new_cell_index, centroid)
        if self.has_cell_shifted_centroid:
            self.set_cell_shifted_centroid(new_cell_index, centroid)
            
        self.set_cell_k(new_cell_index, np.eye(3)*1.e-1)

        print "volume = ", volume
        print "cetnroid =", centroid

        self.apply_neumann_from_function(fracture_boundary_marker, lambda x:np.zeros(3))
        self.output_vtk_faces("new_faces", new_faces)


    def build_mesh(self):
        """ Base class function for constructing the mesh. 
        """
        raise("Has not been implemented")

    

    
    
    
        


