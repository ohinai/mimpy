
import mimpy.mesh.hexmesh as hexmesh
import mimpy.mesh.mesh as mesh
import numpy as np


import unittest

class TestCellDivide(unittest.TestCase):
    
    def test_write(self):
        
        two_cells = hexmesh.HexMesh()
        two_cells.build_mesh(2, 3, 2, 
                             lambda x, i, j, k: np.eye(3), 
                             1., 1., 1.)

        two_cells.build_frac_from_faces([5])
        
        output_file = open("test_output", 'w')
        two_cells.save_mesh(output_file)
        output_file.close()

        input_file = open("test_output")

        loaded_mesh = mesh.Mesh()
        
        loaded_mesh.load_mesh(input_file)
        
        assert(two_cells.get_number_of_faces() ==
               loaded_mesh.get_number_of_faces())

        assert(two_cells.get_number_of_cells() ==
               loaded_mesh.get_number_of_cells())
        
        
        
        
        
        
