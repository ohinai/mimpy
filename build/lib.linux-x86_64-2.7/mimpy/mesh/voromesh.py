from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import math
from . import mesh
from six.moves import map
from six.moves import range
from six.moves import zip


class VoroMesh(mesh.Mesh):
    """ The voro mesh loads data from 
    voro++ voronoi generator by Chris H. Rycroft. 
    The class expects a *.vol file formatted 
    using the following command on voro++:
    voro++ -c "%q$%v$%C$%P$%t$%f$%l" x_min x_max y_min y_max z_min z_max [points_file]
    """
    def build_mesh(self, voro_file_name, K):
        """ Takes permeability tensor function K and 
        voronoi file output from voro++ and builds mesh. 
        """
        done_cells = []
        neighbor_to_face_hash = {}

        voro_file = open(voro_file_name)

        self.set_boundary_markers([0, 1, 2, 3, 4, 5], 
                                  ['BottomX', 'TopX', 
                                   'BottomY', 'TopY',
                                   "BottomZ,", "TopZ",])

        use_shifted_points = True

        if use_shifted_points:
            self.use_face_shifted_centroid()
            self.use_cell_shifted_centroid()


        ## We need a two-pass to get the number of cells first. 
        number_of_cells = 0
        for line in voro_file:
            number_of_cells += 1
            self.add_cell([],[])

        voro_file.seek(0)

        
        shifted_face_out = open("shifted_out", 'w')
        ortho_line_out = open("ortho_line", 'w')

        for line in voro_file:
            full_input = line.split("$")
            
            current_cell = int(full_input[0])

            voronoi_point = full_input[1]
            voronoi_point = np.array([float(point) for point in voronoi_point.split()])

            volume = full_input[2]
            volume = float(volume)

            centroid = full_input[3]
            centroid = np.array([float(point) for point in centroid.split()])

            vertices = full_input[4]
            vertices = vertices.replace(")", "")
            vertices = vertices.replace("(", "")
            vertices = vertices.split()
            vertices = [np.array([float(v) for v in vertex.split(",")]) 
                        for vertex in vertices]

            faces = full_input[5]
            faces = faces.replace(")", "")
            faces = faces.replace("(", "")
            faces = faces.split()
            faces = [[int(f) for f in face.split(",")]
                     for face in faces]

            face_areas = full_input[6]
            face_areas = [float(a) for a in face_areas.split()]

            face_normals = full_input[7]
            face_normals = face_normals.replace(")", "")
            face_normals = face_normals.replace("(", "")
            face_normals = [np.array([float(n) for n in normal.split(",")]) 
                        for normal in face_normals.split()]
            
            for face in faces:
                face.reverse()

            neighbors = full_input[8]
            neighbors = neighbors.split()
            neighbors = [int(n) for n in neighbors]
            
            local_to_global_point_index = [self.add_point(vertex) for vertex in vertices]
            
            current_cell_faces = []
            current_cell_orientations = []

            for (local_face_index, global_cell_index) in enumerate(neighbors):
                if global_cell_index >= 0:
                    if global_cell_index not in done_cells:
                        ## Maps the neighbor cell to the face that was just
                        ## added. This is to avoid doubling the number 
                        ## of faces. 
                        
                        ## We first check that there are no points that are too close 
                        ## to each other. 
                        new_face_points = [local_to_global_point_index[vertex_index]\
                                                            for vertex_index in faces[local_face_index]]
                        
                        points_to_add = []
                        for point in new_face_points:
                            too_close = False
                            for existing_point in points_to_add:
                                if np.linalg.norm(self.get_point(existing_point)-
                                                  self.get_point(point))<1.e-8:
                                    too_close = True

                            if not too_close:
                                points_to_add.append(point)

                        ## exclude faces with less than two points. 
                        if len(points_to_add)>2:
                            new_face_index = self.add_face(points_to_add)

                            self.set_face_normal(new_face_index, face_normals[local_face_index])
                            self.set_face_area(new_face_index,face_areas[local_face_index])

                            self.set_face_real_centroid(new_face_index,
                                                        self.find_face_centroid(new_face_index)[1])

                            neighbor_to_face_hash[(current_cell, global_cell_index)] = new_face_index
                            current_cell_faces.append(new_face_index)
                            current_cell_orientations.append(1)
                        else:
                            pass

                    else:
                        if (global_cell_index, current_cell) in neighbor_to_face_hash:
                            current_cell_faces.append(neighbor_to_face_hash[(global_cell_index, current_cell)])
                            current_cell_orientations.append(-1)
                            if use_shifted_points:
                                self.set_face_shifted_centroid(neighbor_to_face_hash[(global_cell_index, current_cell)], 
                                                               (voronoi_point+
                                                                self.get_cell_shifted_centroid(global_cell_index))*.5)
                                shifted_point = (voronoi_point+\
                                                     self.get_cell_shifted_centroid(global_cell_index))*.5
                                
                        else:
                            pass

                ## Add boundary face
                else:

                    ## We first check that there are no point that are too close 
                    ## to each other. 
                    new_face_points = [local_to_global_point_index[vertex_index]\
                                           for vertex_index in faces[local_face_index]]
                    
                    points_to_add = []
                    for point in new_face_points:
                        too_close = False
                        for existing_point in points_to_add:
                            if np.linalg.norm(self.get_point(existing_point)-
                                              self.get_point(point))<1.e-12:
                                too_close = True
                        if not too_close:
                            points_to_add.append(point)
                    
                    if len(points_to_add)>2:
                        new_face_index = self.add_face(points_to_add)

                        current_cell_faces.append(new_face_index)
                        current_cell_orientations.append(1)

                        self.set_face_normal(new_face_index, face_normals[local_face_index])
                        self.set_face_area(new_face_index, face_areas[local_face_index])
                        self.set_face_real_centroid(new_face_index, self.find_face_centroid(new_face_index)[1])
                    
                        self.add_boundary_face(abs(global_cell_index)-1, new_face_index, 1)

                        if use_shifted_points:
                            self.set_face_shifted_centroid(new_face_index, self.find_face_centroid(new_face_index)[1])

            self.set_cell_faces(current_cell, current_cell_faces)
            self.set_cell_domain(current_cell, 0)
            self.set_cell_orientation(current_cell, current_cell_orientations)
            self.set_cell_volume(current_cell, volume)
            self.set_cell_real_centroid(current_cell, centroid)
            self.set_cell_shifted_centroid(current_cell, voronoi_point)

            done_cells.append(current_cell)
            
        for cell_index in range(self.get_number_of_cells()):
            
            [cx, cy, cz] = self.get_cell_real_centroid(cell_index)
            k_e = K(np.array([cx, cy, cz]))

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

                print(center[0], center[1], end=' ', file=output) 
                print(normal[0] * vectorMagnitude[face], end=' ', file=output) 
                print(normal[1] * vectorMagnitude[face], end=' ', file=output) 
                print(" ", file=output)

    def output_vtk_mesh_w_data(self, filename, CellValues = [], CellValueLabels = []):
        output = open(filename + ".vtk", 'w')
        
        print("# vtk DataFile Version 2.0", file=output)
        print("#unstructured mesh", file=output)
        print("ASCII", file=output)
        print("DATASET UNSTRUCTURED_GRID", file=output)

        print("POINTS", self.get_number_of_points(), "float", file=output)
        
        for p in self.points:
            print(p[0], p[1], p[2], file=output)

        print("CELLS", self.get_number_of_cells(), 9 * self.get_number_of_cells(), file=output)
        
        for cell in self.vtk_cells:
            print(8, " ".join(map(str, cell)), file=output)

        print("CELL_TYPES", self.get_number_of_cells(), file=output)

        for cell in self.vtk_cells:
            print(12, file=output)

        if CellValues:
            print("CELL_DATA", self.get_number_of_cells(), file=output)
            for (entry, entryname) in zip(CellValues, CellValueLabels):
                print("SCALARS", entryname, "double 1", file=output)
                print("LOOKUP_TABLE default", file=output) 
                for value in entry:
                    print(value, file=output)
            
        




