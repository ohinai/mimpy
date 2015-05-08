
import numpy as np
cimport cython 

def all_cell_volumes_centroids(int[:,:] cell_pointer, 
                               int ncells, 
                               int [:] cells, 
                               int [:] cell_normal_orientation,
                               float [:, :] points,
                               float[:] cell_volume, 
                               float[:, :] cell_centroids, 
                               int[:,:] face_pointer, 
                               int nfaces, 
                               int [:] faces, 
                               float [:, :] normals, 
                               int [:, :] face_to_cell, 
                               ):
    """ Computes and sets all the cell volumes and real
    cell centroids for mesh.
    """
    cdef float[:] current_normal
    cdef int[:] current_face
    cdef int face_index 
    cdef int A
    cdef int B
    cdef int C
       
    cdef float P1 
    cdef float Pa 
    cdef float Pb 
    cdef float Paa 
    cdef float Pab 
    cdef float Pbb 

    cdef float a0
    cdef float b0
    
    cdef float a1
    cdef float b1
    
    cdef float da
    cdef float db
    cdef float a0_2
    cdef float a0_3
    cdef float b0_2
    cdef float b0_3
    cdef float a1_2
    cdef float C1
    cdef float Ca
    cdef float Caa
    cdef float Cb
    cdef float Cbb
    cdef float Cab
    cdef float Kab
            

    cdef int len_current_face
    cdef int local_index 

    cdef float[:] first_point

    cdef float w

    for face_index in range(nfaces):
        current_normal = normals[face_index]

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

        current_face = faces[face_pointer[face_index, 0]:
                             face_pointer[face_index, 1]+face_pointer[face_index, 0]]

        len_current_face = face_pointer[face_index, 1]

        for local_index in range(len_current_face):
            current_point = points[current_face[local_index]]
            next_point = points[current_face[(local_index+1)%len_current_face]]

            a0 = current_point[A]
            b0 = current_point[B]
            
            a1 = next_point[A]
            b1 = next_point[B]
            
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

        first_point = points[current_face[0]]

        w = 0.
        w -= current_normal[0]*first_point[0]
        w -= current_normal[1]*first_point[1]
        w -= current_normal[2]*first_point[2]

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
        
        cell_index_1 = face_to_cell[face_index, 0]
        if cell_index_1 >= 0:
            start_index = cell_pointer[cell_index_1, 0]
            end_index = start_index + cell_pointer[cell_index_1, 1]
            current_cell = cells[start_index:end_index]
            current_cell_orientation = cell_normal_orientation[start_index:
                                                                   end_index]

            local_index = -1
            for index in range(cell_pointer[cell_index_1, 1]):
                if current_cell[index] == face_index:
                    local_index = index

            if local_index == -1:
                print "error: Did not find local index during volume calculation"
            
            orientation = current_cell_orientation[local_index]
            if A==0:
                cell_volume[cell_index_1] += current_normal[0]*Fa*orientation
            elif B == 0:
                cell_volume[cell_index_1] += current_normal[0]*Fb*orientation
            else:
                cell_volume[cell_index_1] += current_normal[0]*Fc*orientation
               
            cell_centroids[cell_index_1, A] += current_normal[A]*Faa*orientation
            cell_centroids[cell_index_1, B] += current_normal[B]*Fbb*orientation
            cell_centroids[cell_index_1, C] += current_normal[C]*Fcc*orientation
            
        cell_index_2 = face_to_cell[face_index, 1]
        if cell_index_2 >= 0:
            start_index = cell_pointer[cell_index_2, 0]
            end_index = start_index + cell_pointer[cell_index_2, 1]
            current_cell = cells[start_index:end_index]
            current_cell_orientation = cell_normal_orientation[start_index:
                                                                   end_index]
            local_index = -1
            for index in range(cell_pointer[cell_index_2, 1]):
                if current_cell[index] == face_index:
                    local_index = index
            if local_index == -1:
                print "error: Did not find local index during volume calculation"
            orientation = current_cell_orientation[local_index]
            if A==0:
                cell_volume[cell_index_2] += current_normal[0]*Fa*orientation
            elif B == 0:
                cell_volume[cell_index_2] += current_normal[0]*Fb*orientation
            else:
                cell_volume[cell_index_2] += current_normal[0]*Fc*orientation
               
            cell_centroids[cell_index_2, A] += current_normal[A]*Faa*orientation
            cell_centroids[cell_index_2, B] += current_normal[B]*Fbb*orientation
            cell_centroids[cell_index_2, C] += current_normal[C]*Fcc*orientation
    
    for cell_index in range(ncells):
        cell_centroids[cell_index, 0] /= cell_volume[cell_index]*2.
        cell_centroids[cell_index, 1] /= cell_volume[cell_index]*2.
        cell_centroids[cell_index, 2] /= cell_volume[cell_index]*2.
            
            

