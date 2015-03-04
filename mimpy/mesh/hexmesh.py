
import numpy as np
import math
import mesh
import array 

import hexmesh_cython 

class HexMesh(mesh.Mesh):
    
    def nonplanar_normal(self, face):

        v1 = self.get_point(face[1]) - self.get_point(face[0]) 
        v2 = self.get_point(face[0]) - self.get_point(face[-1]) 
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

    def populate_face_areas(self):
        """ Finds all the faces areas and 
        stores them the area array. 
        """
        hexmesh_cython.all_face_areas(self.faces.pointers, 
                                      len(self.faces), 
                                      self.faces.data, 
                                      self.points, 
                                      self.face_areas)

    def populate_face_normals(self):
        """ Finds all the faces normals and 
        stores them the normals array. 
        """
        hexmesh_cython.all_face_normals(self.faces.pointers, 
                                        len(self.faces), 
                                        self.faces.data, 
                                        self.points, 
                                        self.face_normals)
        
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

        area += np.sqrt(s*(s-a)*(s-b)*(s-c))

        a = np.linalg.norm(p2-p3)
        b = np.linalg.norm(p3-centerPoint)
        c = np.linalg.norm(centerPoint-p2)
        s = (a + b + c)/2.

        area += np.sqrt(s*(s-a)*(s-b)*(s-c))

        a = np.linalg.norm(p3-p4)
        b = np.linalg.norm(p4-centerPoint)
        c = np.linalg.norm(centerPoint - p3)
        s = (a + b + c)/2.

        area += np.sqrt(s*(s-a)*(s-b)*(s-c))

        a = np.linalg.norm(p4-p1)
        b = np.linalg.norm(p1-centerPoint)
        c = np.linalg.norm(centerPoint - p4)
        s = (a + b + c)/2.

        area += np.sqrt(s*(s-a)*(s-b)*(s-c))

        return area 

    def get_dim_x(self):
        """ Return the dimension of the domain 
        in the X direction. 
        """
        return self.dim_x

    def get_dim_y(self):
        """ Return the dimension of the domain 
        in the Y direction. 
        """
        return self.dim_y

    def get_dim_z(self):
        """ Return the dimension of the domain 
        in the Z direction. 
        """
        return self.dim_z

    def build_faces(self, nk, nj, ni):
        """ Function to build the mesh faces.  
        """
        count = 0
        self.polygon_ijka_to_index = {}

        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    if i < ni-1 and j < nj-1: 
                        new_face = array.array('i', [self.ijk_to_index(i,j,k),
                                                     self.ijk_to_index(i+1,j,k),
                                                     self.ijk_to_index(i+1,j+1,k),
                                                     self.ijk_to_index(i,j+1,k)])
                        
                        face_index = self.add_face(new_face)

                        self.set_face_real_centroid(face_index, 
                                                    self.nonplanar_face_centroid(new_face))

                        self.polygon_ijka_to_index[(i,j,k,0)] = face_index
                        if k == 0:
                            self.add_boundary_face(4, face_index, -1)

                        if k == nk-1:
                            self.add_boundary_face(5, face_index, 1)

                        count += 1

                    if k < nk-1 and i < ni-1:
                        new_face = [self.ijk_to_index(i,j,k),
                                    self.ijk_to_index(i,j,k+1), 
                                    self.ijk_to_index(i+1,j,k+1), 
                                    self.ijk_to_index(i+1,j,k)]
                        
                        face_index = self.add_face(new_face)

                        self.set_face_real_centroid(face_index, 
                                                    self.nonplanar_face_centroid(new_face))
                        
                        self.polygon_ijka_to_index[(i,j,k,1)] = face_index

                        if j == 0:
                            self.add_boundary_face(2, face_index, -1)
                            
                        if j == nj - 1:
                            self.add_boundary_face(3, face_index, 1)

                        count += 1

                    if j < nj-1 and k < nk-1:
                        new_face = [self.ijk_to_index(i,j,k),
                                    self.ijk_to_index(i,j+1,k), 
                                    self.ijk_to_index(i,j+1,k+1), 
                                    self.ijk_to_index(i,j,k+1)]

                        face_index = self.add_face(new_face)

                        self.set_face_real_centroid(face_index, 
                                                    self.nonplanar_face_centroid(new_face))

                        self.polygon_ijka_to_index[(i,j,k,2)] = count

                        if i == 0:
                            self.add_boundary_face(0, count, -1)

                        if i == ni - 1:
                            self.add_boundary_face(1, count, 1)

                            
                        count += 1
                        
        self.populate_face_areas()
        self.populate_face_normals()


    def ijk_to_index(i,j, k):
        pass


    def build_mesh(self, ni, nj, nk, K, 
                   dim_x, dim_y, dim_z, 
                   modification_function = lambda x, i, j, k: x):
        
        self.dim = 3

        # Needs to be moved to an __init__ function. 
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z

        self.cell_to_ijk = {}

        dx = dim_x/(ni-1.)
        dy = dim_y/(nj-1.)
        dk = dim_z/(nk-1.)
        
        self.set_boundary_markers([0, 1, 2, 3, 4, 5], 
                                  ['BottomX', 'TopX', 
                                   'BottomY', 'TopY',
                                   "BottomZ,", "TopZ",])

        def ijk_to_index(i,j, k):
            return i+ni*j+k*ni*nj            

        self.ijk_to_index = ijk_to_index

        count = 0
	## adding points:
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):   
                    self.add_point(modification_function(np.array([float(i)*dx,
                                                                   float(j)*dy, 
                                                                   float(k)*dk]), 
                                                         i, j, k))
                    count +=1 
        
        self.build_faces(nk, nj, ni)

	### adding cells:
        for k in range(nk-1):
            for j in range(nj-1):
                for i in range(ni-1):

                    new_cell = array.array('i', [self.polygon_ijka_to_index[(i,j,k, 0)],
                                                 self.polygon_ijka_to_index[(i,j,k, 1)],
                                                 self.polygon_ijka_to_index[(i,j,k, 2)],
                                                 self.polygon_ijka_to_index[(i+1,j,k, 2)],
                                                 self.polygon_ijka_to_index[(i,j+1,k, 1)],
                                                 self.polygon_ijka_to_index[(i,j,k+1, 0)]])
                    
                    cell_index = self.add_cell(new_cell,
                                               array.array('i', [-1, -1, -1, 1, 1, 1]))
                    
                    self.cell_to_ijk[cell_index] = (i, j, k)
                    
        self.find_volume_centroid_all()
                             
        for cell_index in range(self.get_number_of_cells()):
            
            (i, j, k) = self.cell_to_ijk[cell_index]

            [cx, cy, cz] = self.get_cell_real_centroid(cell_index)
            k_e = K(np.array([cx, cy, cz]), i, j, k)

            self.set_cell_k(cell_index, k_e)

    def output_vector_field_gnuplot(self, filename, vectorMagnitude):
        """ For plotting vectors in a 2D plane using gnuplot. 
        The plane is assumed to be x-y. Plot using command:
        plot "filename.dat" using 1:2:3:4 with vectors. 
        """
        output = open(filename +  ".dat", "w")
        
        for face in range(self.get_number_of_faces()):
            if not self.is_boundary_face(face, [0, 1, 2, 3, 4, 5]):
                center = self.get_face_real_centroid(face)
                normal = self.get_face_normal(face)

                print >>output, center[0], center[1], 
                print >>output, normal[0] * vectorMagnitude[face], 
                print >>output, normal[1] * vectorMagnitude[face], 
                print >>output, " "

        






