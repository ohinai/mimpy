#!python
#cython: language_level=3

import numpy as np
cimport cython 

def update_m_fast(double [:] m_coo,
                  double [:] multipliers,
                  double [:] m_data_for_update, 
                  long[:] m_e_locations, 
                  int number_of_cells):
    
    cdef int cell_index
    cdef int m_start
    cdef int m_end
    cdef int local_index

    for cell_index in range(number_of_cells):
        m_start = m_e_locations[cell_index]
        m_end = m_e_locations[cell_index+1]
        
        for local_index in range(m_start, m_end):
            m_coo[local_index] = m_data_for_update[local_index]*multipliers[cell_index]
        
    
    
