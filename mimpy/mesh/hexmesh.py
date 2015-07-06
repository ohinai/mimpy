from __future__ import absolute_import

import numpy as np
import mimpy.mesh.mesh as mesh
import mimpy.mesh.hexmesh_cython as hexmesh_cython
from six.moves import range

class HexMesh(mesh.Mesh):
    """ Class for constructing structured hexahedral meshes.
    """
    def _nonplanar_face_normal(self, face_index):
        """ Calculates an approximate normal for a
        face that might not planar.

        :param int face_index: index of face.
        :return: The face normal.
        :rtype: ndarray
        """
        face = self.get_face(face_index)
        v1 = self.get_point(face[1]) - self.get_point(face[0])
        v2 = self.get_point(face[0]) - self.get_point(face[-1])
        normal = np.cross(v2, v1)

        return normal/np.linalg.norm(normal)

    def _nonplanar_face_centroid(self, face_index):
        """ Calculates an approximate centroid for a
        face that might not planar.

        :param int face_index: index of face.
        :return: Face centroid.
        :rtype: ndarray
        """
        face = self.get_face(face_index)
        p1 = self.get_point(face[0])
        p2 = self.get_point(face[1])
        p3 = self.get_point(face[2])
        p4 = self.get_point(face[3])

        center_point = .25 * (p1 + p2 + p3 + p4)

        return center_point

    def _nonplanar_cell_centroid(self, cell_index):
        """ Calculates an approximate centroid for a
        cell that may have nonplanar faces.
        """
        centroid = np.zeros(3)
        count = 0.
        for face in self.get_cell(cell_index):
            for point in self.get_face(face):
                count += 1.
                centroid += self.get_point(point)

        centroid = centroid/count
        return centroid

    def _populate_face_areas(self):
        """ Finds all the faces areas and
        stores them the area array.
        """
        hexmesh_cython.all_face_areas(self.faces.pointers,
                                      len(self.faces),
                                      self.faces.data,
                                      self.points,
                                      self.face_areas)

    def _populate_face_centroids(self):
        """ Finds all the faces centroids and
        stores them the area array.
        """
        for face_index in range(self.get_number_of_faces()):
            current_centroid = self._nonplanar_face_centroid(face_index)
            self.set_face_real_centroid(face_index, current_centroid)

    def _populate_face_normals(self):
        """ Finds all the faces normals and
        stores them the normals array.
        """
        hexmesh_cython.all_face_normals(self.faces.pointers,
                                        len(self.faces),
                                        self.faces.data,
                                        self.points,
                                        self.face_normals)

    def _nonplanar_face_area(self, face_index):
        """ Calculates an approximate area for a
        face that might not planar.

        :param int face_index: index of face.

        :return: Face area.
        :rtype: float
        """
        face = self.get_face(face_index)
        area = 0.
        p1 = self.points[face[0]]
        p2 = self.points[face[1]]
        p3 = self.points[face[2]]
        p4 = self.points[face[3]]

        center_point = .25 * (p1 + p2 + p3 + p4)

        a = np.linalg.norm(p1-p2)
        b = np.linalg.norm(p2-center_point)
        c = np.linalg.norm(center_point - p1)
        s = (a + b + c)/2.

        area += np.sqrt(s*(s-a)*(s-b)*(s-c))

        a = np.linalg.norm(p2-p3)
        b = np.linalg.norm(p3-center_point)
        c = np.linalg.norm(center_point-p2)
        s = (a + b + c)/2.

        area += np.sqrt(s*(s-a)*(s-b)*(s-c))

        a = np.linalg.norm(p3-p4)
        b = np.linalg.norm(p4-center_point)
        c = np.linalg.norm(center_point - p3)
        s = (a + b + c)/2.

        area += np.sqrt(s*(s-a)*(s-b)*(s-c))

        a = np.linalg.norm(p4-p1)
        b = np.linalg.norm(p1-center_point)
        c = np.linalg.norm(center_point - p4)
        s = (a + b + c)/2.

        area += np.sqrt(s*(s-a)*(s-b)*(s-c))

        return area

    def get_dim_x(self):
        """ Return the dimension of the domain
        in the X direction.

        :return: Domain x dimension.
        :rtype: float
        """
        return self.dim_x

    def get_dim_y(self):
        """ Return the dimension of the domain
        in the Y direction.

        :return: Domain y dimension.
        :rtype: float
        """
        return self.dim_y

    def get_dim_z(self):
        """ Return the dimension of the domain
        in the Z direction.

        :return: Domain z dimension.
        :rtype: float
        """
        return self.dim_z

    def _build_faces(self, ni, nj, nk):
        """ Function to build the mesh faces.

        :param int ni: Number of faces in the x-direction.
        :param int nj: Number of faces in the y-direction.
        :param int nk: Number of faces in the z-direction.

        :return: Dictionary mapping ijka to index.
        :rtype: dict
        """
        count = 0
        polygon_ijka_to_index = {}

        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    if i < ni-1 and j < nj-1:
                        new_face = [self.ijk_to_index(i, j, k),
                                    self.ijk_to_index(i+1, j, k),
                                    self.ijk_to_index(i+1, j+1, k),
                                    self.ijk_to_index(i, j+1, k)]

                        face_index = self.add_face(new_face)

                        polygon_ijka_to_index[(i, j, k, 0)] = face_index
                        if k == 0:
                            self.add_boundary_face(4, face_index, -1)

                        if k == nk-1:
                            self.add_boundary_face(5, face_index, 1)

                        count += 1

                    if k < nk-1 and i < ni-1:
                        new_face = [self.ijk_to_index(i, j, k),
                                    self.ijk_to_index(i, j, k+1),
                                    self.ijk_to_index(i+1, j, k+1),
                                    self.ijk_to_index(i+1, j, k)]

                        face_index = self.add_face(new_face)

                        polygon_ijka_to_index[(i, j, k, 1)] = face_index

                        if j == 0:
                            self.add_boundary_face(2, face_index, -1)

                        if j == nj - 1:
                            self.add_boundary_face(3, face_index, 1)

                        count += 1

                    if j < nj-1 and k < nk-1:
                        new_face = [self.ijk_to_index(i, j, k),
                                    self.ijk_to_index(i, j+1, k),
                                    self.ijk_to_index(i, j+1, k+1),
                                    self.ijk_to_index(i, j, k+1)]

                        face_index = self.add_face(new_face)

                        polygon_ijka_to_index[(i, j, k, 2)] = count

                        if i == 0:
                            self.add_boundary_face(0, count, -1)

                        if i == ni - 1:
                            self.add_boundary_face(1, count, 1)

                        count += 1

        self._populate_face_areas()
        self._populate_face_centroids()
        self._populate_face_normals()

        return polygon_ijka_to_index

    def ijk_to_index(i, j, k):
        """ Returns cell index number for an i, j, k numbering.

        :param int i: index in x-direction.
        :param int j: index in y-direction.
        :param int k: index in z-direction.
        """
        pass

    def __init__(self):
        """ Initialize hexmesh.
        """
        mesh.Mesh.__init__(self)
        self.dim_x = 0.0
        self.dim_y = 0.0
        self.dim_z = 0.0

        self.cell_to_ijk = {}

    def build_mesh(self, ni, nj, nk,
                   dim_x, dim_y, dim_z, K, 
                   modification_function = None):
        """ Constructs a structured hexahedral mesh.

        :param int ni: Number of cells in the x-direction.
        :param int nj: Number of cells in the y-direction.
        :param int nk: Number of cells in the z-direction.
        :param function K: Permeability map function.
             K(point, i, j, k ) -> 3x3 Matrix.
        :param float dim_x: Size of domain in the x-direction.
        :param float dim_y: Size of domain in the y-direction.
        :param float dim_z: Size of domain in the z-direction.
        :param function  modification_function: Alteration function for shifting
              points of the cells. modification_function(p) -> 3 array.

        :return: None
        """
        # Needs to be moved to an __init__ function.
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z

        ni += 1
        nj += 1
        nk += 1

        dx = self.dim_x/float(ni-1.)
        dy = self.dim_y/float(nj-1.)
        dz = self.dim_z/float(nk-1.)

        self.set_boundary_markers([0, 1, 2, 3, 4, 5],
                                  ['BottomX', 'TopX',
                                   'BottomY', 'TopY',
                                   "BottomZ,", "TopZ",])

        def ijk_to_index(i, j, k):
            return i+ni*j+k*ni*nj

        self.ijk_to_index = ijk_to_index

        ## adding points:
        if modification_function == None:
            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        self.add_point(np.array([float(i)*dx,
                                                 float(j)*dy,
                                                 float(k)*dz]))
        else:
            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        self.add_point(modification_function(
                                np.array([float(i)*dx,
                                          float(j)*dy,
                                          float(k)*dz]), i, j, k))

        polygon_ijka_to_index = self._build_faces(ni, nj, nk)

	### adding cells:
        for k in range(nk-1):
            for j in range(nj-1):
                for i in range(ni-1):
                    new_cell = [polygon_ijka_to_index[(i, j, k, 0)],
                                polygon_ijka_to_index[(i, j, k, 1)],
                                polygon_ijka_to_index[(i, j, k, 2)],
                                polygon_ijka_to_index[(i+1, j, k, 2)],
                                polygon_ijka_to_index[(i, j+1, k, 1)],
                                polygon_ijka_to_index[(i, j, k+1, 0)]]

                    cell_index = self.add_cell(new_cell,
                                               [-1, -1, -1, 1, 1, 1])

                    self.cell_to_ijk[cell_index] = (i, j, k)

        self.find_volume_centroid_all()
        for cell_index in range(self.get_number_of_cells()):
            (i, j, k) = self.cell_to_ijk[cell_index]
            [cx, cy, cz] = self.get_cell_real_centroid(cell_index)
            k_e = K(np.array([cx, cy, cz]), i, j, k)

            self.set_cell_k(cell_index, k_e)


