#!python
#cython: language_level=3

import numpy as np
cimport cython 
cdef extern from "math.h":
    double sqrt(double n)



cdef dist(double[:] v1, double[:] v2):
    cdef double summation = (v1[0]-v2[0])*(v1[0]-v2[0])
    summation += (v1[1]-v2[1])*(v1[1]-v2[1])
    summation += (v1[2]-v2[2])*(v1[2]-v2[2])
    return sqrt(summation)


cdef cross(double[:] v1, double[:] v2, double[:] result):
    result[0] = v1[1]*v2[2] - v1[2]*v2[1]
    result[1] = v1[2]*v2[0] - v1[0]*v2[2]
    result[2] = v1[0]*v2[1] - v1[1]*v2[0]

cdef normalize(double[:] v):
    normal=v[0]*v[0]
    normal+=v[1]*v[1]
    normal+=v[2]*v[2]
    normal = sqrt(normal)
    v[0] /= normal
    v[1] /= normal
    v[2] /= normal

def all_face_areas(int[:,:] pointer, 
                   int nfaces, 
                   int [:] faces, 
                   double [:, :] points,
                   double [:] face_areas):

    cdef int start_point 
    cdef double[:] p1 
    cdef double[:] p2 
    cdef double[:] p3 
    cdef double[:] p4 
    cdef double[:] center_point = np.array([0., 0., 0.], dtype=np.dtype('d'))

    cdef double a 
    cdef double b
    cdef double c
    cdef double s
    cdef double area

    cdef int face_index 
    for face_index in range(nfaces):
        start_point = pointer[face_index, 0]
        p1 = points[faces[start_point]]
        p2 = points[faces[start_point+1]]
        p3 = points[faces[start_point+2]]
        p4 = points[faces[start_point+3]]

        center_point[0] = .25 * (p1[0] + p2[0] + p3[0] + p4[0])
        center_point[1] = .25 * (p1[1] + p2[1] + p3[1] + p4[1])
        center_point[2] = .25 * (p1[2] + p2[2] + p3[2] + p4[2])

        area = 0
        a = dist(p1,p2)
        b = dist(p2, center_point)
        c = dist(center_point, p1)
        s = (a + b + c)/2.
        
        area += sqrt(s*(s-a)*(s-b)*(s-c))
        
        a = dist(p2, p3)
        b = dist(p3,center_point)
        c = dist(center_point, p2)
        s = (a + b + c)/2.
        
        area += sqrt(s*(s-a)*(s-b)*(s-c))
        
        a = dist(p3, p4)
        b = dist(p4, center_point)
        c = dist(center_point, p3)
        s = (a + b + c)/2.
        
        area += sqrt(s*(s-a)*(s-b)*(s-c))
        
        a = dist(p4, p1)
        b = dist(p1, center_point)
        c = dist(center_point, p4)
        s = (a + b + c)/2.
        
        area += sqrt(s*(s-a)*(s-b)*(s-c))
        face_areas[face_index] = area

def all_face_normals(int[:,:] pointer, 
                     int nfaces, 
                     int [:] faces, 
                     double[:, :] points,
                     double[:, :] face_normals):
    
    
    cdef double[3] v1
    cdef double[3] v2
    cdef double[:] p1
    cdef double[:] p2
    cdef double[:] p3
    
    for face_index in range(nfaces):
        p1 = points[faces[pointer[face_index, 0]]]
        p2 = points[faces[pointer[face_index, 0]+1]]
        p3 = points[faces[pointer[face_index, 0]+2]]
        
        v2[0] = p2[0] - p3[0]
        v2[1] = p2[1] - p3[1]
        v2[2] = p2[2] - p3[2]

        v1[0] = p1[0] - p2[0]
        v1[1] = p1[1] - p2[1]
        v1[2] = p1[2] - p2[2]

        cross(v1, v2, face_normals[face_index])
        normalize(face_normals[face_index])
        
