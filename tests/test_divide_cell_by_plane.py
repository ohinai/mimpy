import unittest

import mimpy.mesh.hexmesh as hexmesh
import numpy as np


class TestCellDivide(unittest.TestCase):
    def test_single_hex_cell_1(self):
        normals = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            -np.array([1.0, 0.0, 0.0]),
            -np.array([0.0, 1.0, 0.0]),
            -np.array([0.0, 0.0, 1.0]),
        ]

        for normal in normals:
            single_cell = hexmesh.HexMesh()
            single_cell.build_mesh(1, 1, 1, 1.0, 1.0, 1.0, lambda x, i, j, k: np.eye(3))
            single_cell.divide_cell_by_plane(0, np.array([0.5, 0.5, 0.5]), normal)

            assert single_cell.get_number_of_cells() == 2
            assert abs(single_cell.get_cell_volume(0) - 0.5) < 1.0e-12
            assert abs(single_cell.get_cell_volume(1) - 0.5) < 1.0e-12

    def test_single_hex_cell_2(self):
        normals = [
            np.array([1.2, 0.8, 0.0]),
            np.array([1.2, 0.2, 1.3]),
            -np.array([1.2, 0.8, 0.0]),
            -np.array([1.2, 0.2, 1.3]),
        ]

        for normal in normals:
            single_cell = hexmesh.HexMesh()
            single_cell.build_mesh(1, 1, 1, 1.0, 1.0, 1.0, lambda x, i, j, k: np.eye(3))
            single_cell.divide_cell_by_plane(0, np.array([0.5, 0.5, 0.5]), normal)

            assert single_cell.get_number_of_cells() == 2

    def test_single_hex_cell_3(self):
        normals = [
            np.array([1.2, 0.8, 0.0]),
            np.array([1.2, 0.2, 1.3]),
            -np.array([1.2, 0.8, 0.0]),
            -np.array([1.2, 0.2, 1.3]),
        ]

        for normal in normals:
            single_cell = hexmesh.HexMesh()
            single_cell.build_mesh(3, 3, 3, 1.0, 1.0, 1.0, lambda x, i, j, k: np.eye(3))
            cell_index = single_cell.find_cell_near_point(np.array([0.5, 0.5, 0.5]))
            single_cell.divide_cell_by_plane(
                cell_index, np.array([0.5, 0.5, 0.5]), normal
            )
            assert single_cell.get_number_of_cells() == 28
