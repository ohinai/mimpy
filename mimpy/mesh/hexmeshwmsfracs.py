import array
import math

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon

from .hexmesh import *


def three_d_line_line_intersection(a1, a2, b1, b2):
    ts = np.linalg.lstsq(np.array([a2 - a1, -b2 + b1]).T, b1 - a1)[0]

    a_intersect = a1 + (a2 - a1) * ts[0]
    b_intersect = b1 + (b2 - b1) * ts[1]

    if np.linalg.norm(a_intersect - b_intersect) > 1.0e-5:
        return (None, [-1, -1])

    return (a_intersect, ts)


class FracData:
    def __init__(self):
        self.points = []
        self.face = []

        self.center = np.zeros(3)
        self.normal = None
        self.gnu_out = open("test_i", "w")
        self.dip = 0.0
        self.azimuth = 0.0
        self.a = 1.0
        self.b = 1.0

    def output_vtk(self, file_name):
        output = open(file_name + ".vtk", "w")
        print("# vtk DataFile Version 2.0", file=output)
        print("# unstructured mesh", file=output)
        print("ASCII", file=output)
        print("DATASET UNSTRUCTURED_GRID", file=output)
        print("POINTS", len(self.points), "float", file=output)

        for point_index in range(len(self.points)):
            point = self.points[point_index]
            print(point[0], point[1], point[2], file=output)

        print("CELLS", 1, file=output)
        print(len(self.points) + 1, file=output)

        print(len(self.points), end=" ", file=output)
        for point_index in self.face:
            print(point_index, end=" ", file=output)
        print("\n", file=output)

        print("CELL_TYPES", 1, file=output)
        print(7, file=output)

        output.close()

    def build_rotation_matrix(self):
        theta = np.pi / 2.0 - self.azimuth

        u = np.array([np.sin(theta), np.cos(theta), 0.0])

        R = np.zeros((3, 3))
        u = u / np.linalg.norm(u)
        c_t = np.cos(self.dip)
        s_t = np.sin(self.dip)

        u_x = u[0]
        u_y = u[1]
        u_z = u[2]

        R[0, 0] = c_t + u_x**2 * (1.0 - c_t)
        R[0, 1] = u_x * u_y * (1.0 - c_t) - u_z * s_t
        R[0, 2] = u_x * u_z * (1.0 - c_t) + u_y * s_t

        R[1, 0] = u_y * u_x * (1.0 - c_t) + u_z * s_t
        R[1, 1] = c_t + u_y**2 * (1.0 - c_t)
        R[1, 2] = u_y * u_z * (1.0 - c_t) - u_x * s_t

        R[2, 0] = u_z * u_x * (1.0 - c_t) - u_y * s_t
        R[2, 1] = u_z * u_y * (1.0 - c_t) + u_x * s_t
        R[2, 2] = c_t + u_z**2 * (1.0 - c_t)

        return R

    def generate_polygon(self, level):
        poly_out = open("ellips", "w")

        theta = np.pi / 2.0 - self.azimuth
        phi = self.dip

        frac_normal = self.get_normal()

        frac_normal /= np.linalg.norm(frac_normal)

        print(self.center[0], self.center[1], self.center[2], file=poly_out)
        print(self.center[0] + frac_normal[0], end=" ", file=poly_out)
        print(self.center[1] + frac_normal[1], end=" ", file=poly_out)
        print(self.center[2] + frac_normal[2], file=poly_out)
        print("\n", file=poly_out)

        R = self.build_rotation_matrix()

        for t in np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / float(level)):
            x = self.a * np.cos(t)
            y = self.b * np.sin(t)

            theta = np.pi / 2.0 - self.azimuth
            # Add azimuth
            X_p = x * np.cos(theta) + y * np.sin(theta)
            Y_p = -x * np.sin(theta) + y * np.cos(theta)

            dipped = R.dot(np.array([X_p, Y_p, 0.0]))

            [X, Y, Z] = dipped + self.center

            print(X, Y, Z, file=poly_out)

            self.points.append(np.array([X, Y, Z]))
            self.face.append(len(self.points) - 1)

    def divide_polygon_for_intersection(self, segments):
        """Generates multiple polygons based on cutting the
        fracture faces by line segments.
        """
        R = self.build_rotation_matrix()
        fracture_poly_list = []
        # face_poly_list = []

        for point in self.points:
            rot_point = np.linalg.solve(R, point - self.center)
            fracture_poly_list.append(rot_point[:2])

        seg_rot = []
        for seg in segments:
            p1 = seg[0]
            p2 = seg[1]

            vec = p2 - p1
            p1 = p1 - 100.0 * vec
            p2 = p2 + 100.0 * vec

            p1_rot = np.linalg.solve(R, p1 - self.center)
            p2_rot = np.linalg.solve(R, p2 - self.center)

            line = LineString((p1_rot[:2], p2_rot[:2]))
            dilated = line.buffer(1.0e-10)
            seg_rot.append(dilated)

        fracture_poly = Polygon(fracture_poly_list)

        divided_polygons = fracture_poly.difference(seg_rot[0])

        return (divided_polygons, fracture_poly)

    def is_point_on_frac_old(self, p):
        """Returns True if point lies on the
        fracture.
        """
        x_i = p[0]
        y_i = p[1]
        z_i = p[2]

        x_f = self.center[0]
        y_f = self.center[1]
        z_f = self.center[2]

        a = self.a
        b = self.b

        sum_1 = (x_i - x_f) * np.cos(theta)
        sum_1 += (y_i - y_f) * np.sin(theta)
        sum_1 = sum_1**2
        sum_1 /= a**2
        sum_1 /= np.cos(phi) ** 2

        sum_2 = -(x_i - x_f) * np.sin(theta)
        sum_2 += (y_i - y_f) * np.cos(theta)
        sum_2 = sum_2**2
        sum_2 /= b**2

        if sum_1 + sum_2 - 1 < 0.0:
            return True

        return False

    def is_point_on_frac(self, p):
        """Returns True if point lies on the
        fracture.
        """
        R = self.build_rotation_matrix()

        theta = np.pi / 2.0 - self.azimuth
        phi = self.dip

        p_prime = np.linalg.solve(R, p - self.center)

        az_rot = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )

        if abs(p_prime[2]) < 1.0e-8:
            p_prime = np.linalg.solve(az_rot, p_prime[:2])
            x = p_prime[0]
            y = p_prime[1]

            sum_1 = x**2 / self.a**2
            sum_1 += y**2 / self.b**2
            if sum_1 < 1.0:
                return True
        return False

        x_i = p[0]
        y_i = p[1]
        z_i = p[2]

        x_f = self.center[0]
        y_f = self.center[1]
        z_f = self.center[2]

        a = self.a
        b = self.b

        theta = np.pi / 2.0 - self.azimuth
        phi = self.dip

        distance = (x_i - x_f) * np.sin(phi) * np.cos(theta)
        distance += -(y_i - y_f) * np.sin(phi) * np.sin(theta)
        distance += (z_i - z_f) * np.cos(phi)

        X_i = x_i - distance * np.sin(phi) * np.cos(theta)
        Y_i = y_i - distance * np.sin(phi) * np.cos(theta)
        Z_i = z_i - distance * np.cos(phi)

        sum_1 = (X_i - x_f) * np.cos(theta)
        sum_1 += (Y_i - y_f) * np.sin(theta)
        sum_1 = sum_1**2
        sum_1 /= a**2
        sum_1 /= np.cos(phi) ** 2

        sum_2 = -(X_i - x_f) * np.sin(theta)
        sum_2 += (Y_i - y_f) * np.cos(theta)
        sum_2 = sum_2**2
        sum_2 /= b**2

        if sum_1 + sum_2 - 1.0 < 0.0:
            return True

        return False

    def get_normal(self):
        theta = np.pi / 2.0 - self.azimuth
        phi = self.dip

        normal = np.array(
            [np.sin(phi) * np.cos(theta), -np.sin(phi) * np.sin(theta), np.cos(phi)]
        )
        normal /= np.linalg.norm(normal)
        return normal

    def line_frac_boundary_intercept(self, p1, p2):
        p1_frac = self.points[self.face[0]]
        for frac_point in self.face[1:] + [self.face[0]]:
            p2_frac = self.points[frac_point]

            ts = np.linalg.lstsq(
                np.array([p1 - p2, -p2_frac + p1_frac]).T, p1_frac - p1
            )[0]

            if 0.0 < ts[0] < 1.0:
                print(p1[0], p1[1], p1[2], file=self.gnu_out)
                print(p2[0], p2[1], p2[2], file=self.gnu_out)

                print("\n", file=self.gnu_out)

                print(p1_frac[0], p1_frac[1], p1_frac[2], file=self.gnu_out)
                print(p2_frac[0], p2_frac[1], p2_frac[2], file=self.gnu_out)

                print("\n", file=self.gnu_out)

            p1_frac = self.points[frac_point]

    def find_frac_frac_line(self, other_frac):
        """Returns line segments of intersecting two fractures."""

        intersection_points = []

        on_frac = False

        plane_normal = other_frac.normal
        point_on_plane = other_frac.center

        p1_frac = self.points[self.face[0]]
        for frac_point in self.face[1:] + [self.face[0]]:
            p2_frac = self.points[frac_point]

            vector = p2_frac - p1_frac
            vector /= np.linalg.norm(vector)

            d = np.dot((point_on_plane - p1_frac), plane_normal)
            denom = np.dot(vector, plane_normal)

            length = np.linalg.norm(p1_frac - p2_frac)

            if abs(denom) < 1e-10:
                pass
            else:
                d /= denom
                if d <= length + 1.0e-8 and d > 0.0 + 1.0e-8:
                    intersection_point = d * vector + p1_frac
                    intersection_points.append(intersection_point)

                    if other_frac.is_point_on_frac(intersection_point):
                        on_frac = True

            p1_frac = self.points[frac_point]

        if on_frac:
            return intersection_points

        return None


class HexMeshWMSFracs(HexMesh):
    def __init__(self):
        HexMesh.__init__(self)
        self.fracture_faces = set()
        self.fracture_faces_multi = {}

    def build_mesh(
        self,
        ni,
        nj,
        nk,
        K,
        dim_x,
        dim_y,
        dim_z,
        modification_function=lambda x, i, j, k: x,
    ):
        HexMesh.build_mesh(
            self, ni, nj, nk, K, dim_x, dim_y, dim_z, modification_function
        )

        self.intersecting_cells = [0] * self.get_number_of_cells()
        ## Takes the face index and maps it to the fracture.
        self.face_to_fracture = {}
        self.fracture_to_face = {}

    def add_point_to_other_faces(self, faces, point):
        """Adds point to faces in cell. To be used
        when slicing a cell, it adds points that already
        colinear with line segments for existing faces.
        """
        for face_index in faces:
            current_face = self.get_face(face_index)
            current_point = self.get_point(current_face[-1])
            new_face = []
            found_intersection = False
            for point_index in current_face:
                next_point = self.get_point(point_index)

                vec1 = next_point - current_point
                length = np.linalg.norm(vec1)
                vec1 /= np.linalg.norm(vec1)
                vec2 = point - current_point

                # if abs(np.linalg.norm(vec2)-np.dot(vec2, vec1)) < 1.e-8:
                #    print(np.linalg.norm(vec2), abs(np.linalg.norm(vec2)-np.dot(vec2, vec1)))
                if (
                    np.linalg.norm(vec2) > 1.0e-8
                    and np.linalg.norm(vec2) < length
                    and np.linalg.norm(point - next_point) > 1.0e-8
                    and abs(np.linalg.norm(vec2) - np.dot(vec2, vec1)) < 1.0e-8
                ):
                    new_face.append(self.add_point(point))
                    new_face.append(point_index)
                    found_intersection = True
                else:
                    new_face.append(point_index)

                current_point = next_point

            if found_intersection:
                self.set_face(face_index, new_face)

    def construct_polygon_from_segments(self, segments):
        """Takes point pairs and constructs a single polygon
        from joining all the ends. The pairs are identified
        by directly comparing the point locations.
        """
        ## Start by setting the first point.

        current_segments = list(segments)

        new_face = [current_segments[0][0]]
        point_to_match = current_segments[0][1]

        current_segments.pop(0)

        while len(current_segments) > 0:
            to_be_removed = None
            hits = 0
            for index, segment in enumerate(current_segments):
                if (
                    np.linalg.norm(
                        self.get_point(point_to_match) - self.get_point(segment[0])
                    )
                    < 1.0e-8
                ):
                    new_face.append(segment[0])
                    to_be_removed = index
                    next_point_to_match = segment[1]
                    hits += 1
                elif (
                    np.linalg.norm(
                        self.get_point(point_to_match) - self.get_point(segment[1])
                    )
                    < 1.0e-8
                ):
                    new_face.append(segment[1])
                    to_be_removed = index
                    next_point_to_match = segment[0]
                    hits += 1

            try:
                assert hits == 1
            except:
                for seg in segments:
                    print(self.get_point(seg[0]), self.get_point(seg[1]))
                1 / 0

            # print("took out assert == 1")
            current_segments.pop(to_be_removed)
            point_to_match = next_point_to_match

        return new_face

    def get_fracture_intersection_diff(self, fracture, face):
        R = fracture.build_rotation_matrix()

        fracture_poly_list = []
        face_poly_list = []
        for point in fracture.points:
            rot_point = np.linalg.solve(R, point - fracture.center)
            fracture_poly_list.append(rot_point[:2])

        for point_index in face:
            point = self.get_point(point_index)
            rot_point = np.linalg.solve(R, point - fracture.center)
            face_poly_list.append(rot_point[:2])

        face_poly = Polygon(face_poly_list)
        fracture_poly = Polygon(fracture_poly_list)

        if not face_poly.intersects(fracture_poly) or fracture_poly.contains(face_poly):
            return (False, None, None)

        else:
            poly1 = fracture_poly.intersection(face_poly)
            poly1_ret = []
            for point in list(poly1.exterior.coords)[:-1]:
                rot_point = R.dot(np.array(list(point) + [0.0])) + fracture.center
                poly1_ret.append(rot_point)

            poly2 = face_poly.difference(fracture_poly)
            poly2_ret = []
            if type(poly2) == type(MultiPolygon()):
                for poly in poly2:
                    poly2_ret.append([])
                    for point in list(poly.exterior.coords)[:-1]:
                        rot_point = (
                            R.dot(np.array(list(point) + [0.0])) + fracture.center
                        )
                        poly2_ret[-1].append(rot_point)

            else:
                poly2_ret.append([])
                for point in list(poly2.exterior.coords)[:-1]:
                    rot_point = R.dot(np.array(list(point) + [0.0])) + fracture.center
                    poly2_ret[-1].append(rot_point)

            return (True, [poly1_ret], poly2_ret)

    def get_fracture_intersection_diff_w_segs(self, fracture, segs, face):
        (divided_polygons, full_frac_poly) = fracture.divide_polygon_for_intersection(
            segs
        )

        R = fracture.build_rotation_matrix()

        face_poly_list = []
        for point_index in face:
            point = self.get_point(point_index)
            rot_point = np.linalg.solve(R, point - fracture.center)
            face_poly_list.append(rot_point[:2])

        face_poly = Polygon(face_poly_list)

        if not face_poly.intersects(divided_polygons) or full_frac_poly.contains(
            face_poly
        ):
            return (False, None, None)

        else:
            new_faces = []
            for poly in divided_polygons:
                poly1 = poly.intersection(face_poly)
                poly1_ret = []
                if not poly1.is_empty:
                    for point in list(poly1.exterior.coords)[:-1]:
                        rot_point = (
                            R.dot(np.array(list(point) + [0.0])) + fracture.center
                        )
                        poly1_ret.append(rot_point)
                    new_faces.append(poly1_ret)

            poly2 = face_poly.difference(full_frac_poly)
            poly2_ret = []
            if type(poly2) == type(MultiPolygon()):
                for poly in poly2:
                    poly2_ret.append([])
                    for point in list(poly.exterior.coords)[:-1]:
                        rot_point = (
                            R.dot(np.array(list(point) + [0.0])) + fracture.center
                        )
                        poly2_ret[-1].append(rot_point)

            else:
                poly2_ret.append([])
                for point in list(poly2.exterior.coords)[:-1]:
                    rot_point = R.dot(np.array(list(point) + [0.0])) + fracture.center
                    poly2_ret[-1].append(rot_point)

            return (True, new_faces, poly2_ret)

    def add_fracture_meshes(self, fracture_mesh):
        """Couples the fracture mesh with the internal
        boundaries already calculated using add_fractures.
        """
        list_of_faces = []
        list_of_cells = []

        ## Add fracture cells to reservoir mesh.

        frac_point_to_res_point = {}

        for point_index in range(fracture_mesh.get_number_of_points()):
            new_point_index = self.add_point(fracture_mesh.get_point(point_index))
            frac_point_to_res_point[point_index] = new_point_index

        frac_face_to_res_face = {}

        for face_index in range(fracture_mesh.get_number_of_faces()):
            frac_face = fracture_mesh.get_face(face_index)
            res_face = [
                frac_point_to_res_point[point_index] for point_index in frac_face
            ]
            new_face_index = self.add_face(res_face)
            frac_face_to_res_face[face_index] = new_face_index

            self.set_face_real_centroid(
                new_face_index, fracture_mesh.get_face_real_centroid(face_index)
            )

            self.set_face_area(new_face_index, fracture_mesh.get_face_area(face_index))
            self.set_face_normal(
                new_face_index, fracture_mesh.get_face_normal(face_index)
            )

        frac_cell_to_res_cell = {}
        for cell_index in range(fracture_mesh.get_number_of_cells()):
            frac_cell = fracture_mesh.get_cell(cell_index)
            frac_orientations = fracture_mesh.get_cell_normal_orientation(cell_index)

            res_cell = [frac_face_to_res_face[face_index] for face_index in frac_cell]

            new_cell_index = self.add_cell(res_cell, frac_orientations, cell_domain=1)

            self.set_cell_real_centroid(
                new_cell_index, fracture_mesh.get_cell_real_centroid(cell_index)
            )
            self.set_cell_volume(
                new_cell_index, fracture_mesh.get_cell_volume(cell_index)
            )
            self.set_cell_k(new_cell_index, fracture_mesh.get_cell_k(cell_index))

            frac_cell_to_res_cell[cell_index] = new_cell_index

            self.intersecting_cells.append(0)

        frac_cell_to_res_face = {}
        frac_cell_to_res_orientation = {}

        for face_index in self.fracture_faces:
            fracture_id = self.face_to_fracture[face_index]

            for frac_cell_index in fracture_mesh.fracture_id_to_cell[fracture_id]:
                centroid = self.get_face_real_centroid(face_index)
                if fracture_mesh.is_point_on_fracture_face(frac_cell_index, centroid):
                    list_of_faces.append(face_index)
                    list_of_cells.append(cell_index)

                    cell_index = frac_cell_to_res_cell[frac_cell_index]

                    if len(self.get_face_to_cell(face_index)) == 2:
                        bot_res_face_index = self.add_face(
                            list(self.get_face(face_index))
                        )
                        self.set_face_area(
                            bot_res_face_index, self.get_face_area(face_index)
                        )
                        self.set_face_normal(
                            bot_res_face_index, self.get_face_normal(face_index)
                        )
                        self.set_face_real_centroid(
                            bot_res_face_index, self.get_face_real_centroid(face_index)
                        )
                        if self.has_face_shifted_centroid:
                            self.set_face_shifted_centroid(
                                bot_res_face_index,
                                self.get_face_real_centroid(face_index),
                            )

                        bottom_cell = self.get_face_to_cell(face_index)[1]

                        new_cell_faces = array.array("i", self.get_cell(bottom_cell))
                        local_face_index_in_cell = list(new_cell_faces).index(
                            face_index
                        )

                        new_cell_faces[local_face_index_in_cell] = bot_res_face_index

                        top_cell_index = self.get_face_to_cell(face_index)[0]
                        local_top_face_index_in_cell = list(
                            self.get_cell(top_cell_index)
                        ).index(face_index)
                        top_res_face_orientation = self.get_cell_normal_orientation(
                            top_cell_index
                        )[local_top_face_index_in_cell]

                        # self.face_to_cell[face_index].remove(bottom_cell)
                        self.remove_from_face_to_cell(face_index, bottom_cell)
                        self.set_cell_faces(bottom_cell, new_cell_faces)

                    if cell_index in frac_cell_to_res_face:
                        frac_cell_to_res_face[cell_index] += [
                            face_index,
                            bot_res_face_index,
                        ]
                        frac_cell_to_res_orientation[cell_index] += [
                            top_res_face_orientation,
                            -top_res_face_orientation,
                        ]

                    else:
                        frac_cell_to_res_face[cell_index] = [
                            face_index,
                            bot_res_face_index,
                        ]
                        frac_cell_to_res_orientation[cell_index] = [
                            top_res_face_orientation,
                            -top_res_face_orientation,
                        ]

                    self.set_dirichlet_face_pointer(
                        face_index, top_res_face_orientation, cell_index
                    )

                    self.set_dirichlet_face_pointer(
                        bot_res_face_index, -top_res_face_orientation, cell_index
                    )

        for cell_index in frac_cell_to_res_face:
            self.set_forcing_pointer(
                cell_index,
                frac_cell_to_res_face[cell_index],
                frac_cell_to_res_orientation[cell_index],
            )

        ## Apply no-flow boundary conditions to the fracture.

        fracture_boundary_marker = self.create_new_boundary_marker("fracture_boundary")
        for frac_face_index, orientation in fracture_mesh.get_boundary_faces_by_marker(
            0
        ):
            face_index = frac_face_to_res_face[frac_face_index]
            self.add_boundary_face(fracture_boundary_marker, face_index, orientation)

            # self.set_face_quadrature_points(face_index,
            #                                fracture_mesh.get_face_quadrature_points(frac_face_index))

            # self.set_face_quadrature_weights(face_index,
            #                                fracture_mesh.get_face_quadrature_weights(frac_face_index))

        self.apply_neumann_from_function(
            fracture_boundary_marker, lambda x: np.array([0.0, 0.0, 0.0])
        )

        self.output_vtk_faces(
            "internal_faces_", list_of_faces, [list_of_cells], ["CELLS"]
        )

    def add_fractures(self, fracture, segments=[]):
        point_on_plane = fracture.center
        plane_normal = fracture.normal

        self.fracture_to_face[fracture.id] = []
        polygon_out = open("split_polygon.dat", "w")

        clipping_out = open("clippping", "w")
        clipping_2_out = open("clippping2", "w")

        new_points = []

        fracture_faces = set()

        current_key = len(self.fracture_faces_multi)
        self.fracture_faces_multi[current_key] = set()

        face_segments_to_be_added = {}
        done_faces = []

        cells_to_divide = set()
        cells_on_fracture = set()
        gnu_1 = open("in_points", "w")
        gnu_2 = open("out_points", "w")

        for cell_index in range(self.get_number_of_cells()):
            for face_index in self.get_cell(cell_index):
                face = self.get_face(face_index)
                face_offset = list(face[1:]) + [face[0]]

                for point_index, next_point_index in zip(face, face_offset):
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

                        length = np.linalg.norm(p1 - p2)
                        if d <= length + 1.0e-8 and d > 0.0 + 1.0e-8:
                            intersection_point = d * vector + p1
                            if fracture.is_point_on_frac(intersection_point):
                                cells_to_divide.add(cell_index)
                                cells_on_fracture.add(cell_index)
                                self.intersecting_cells[cell_index] = 1
                                print(
                                    intersection_point[0],
                                    intersection_point[1],
                                    intersection_point[2],
                                    file=gnu_2,
                                )
                            else:
                                print(
                                    intersection_point[0],
                                    intersection_point[1],
                                    intersection_point[2],
                                    file=gnu_1,
                                )

                ## Probably don't need to divide entire cell, but just the
                ## faces on the boundary.
                for seg in segments:
                    p1 = seg[0]
                    p2 = seg[1]

                    if self.is_line_seg_intersect_face(face_index, p1, p2):
                        cells_to_divide.add(cell_index)
                        if self.intersecting_cells[cell_index] != 1:
                            self.intersecting_cells[cell_index] = 2

        for cell_index in cells_to_divide:
            current_cell = self.get_cell(cell_index)

            interior_face_segments = []

            faces_left = list(current_cell)
            faces_left = [x for x in faces_left if x not in done_faces]

            ## Face dividing should use general Shapely
            ## polygon splitting.
            for face_index in faces_left:
                done_faces.append(face_index)
                face = self.get_face(face_index)

                new_face_1 = []
                new_face_2 = []

                face_offset = list(face[1:]) + [face[0]]
                intersection_switch = True
                for point_index, next_point_index in zip(face, face_offset):
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
                        length = np.linalg.norm(p1 - p2)
                        if d <= length + 1.0e-8 and d > 0.0 - 1.0e-8:
                            new_point_index = self.add_point(d * vector + p1)
                            new_face_1.append(new_point_index)
                            new_face_2.append(new_point_index)
                            if intersection_switch:
                                interior_face_segments.append([new_point_index])
                            else:
                                interior_face_segments[-1].append(new_point_index)
                            intersection_switch = not intersection_switch

                if len(new_face_2) > 0:
                    self.set_face(face_index, new_face_1)
                    assert len(new_face_1) > 2
                    (face_1_area, face_1_centroid) = self.find_face_centroid(face_index)
                    self.set_face_real_centroid(face_index, face_1_centroid)
                    self.set_face_area(face_index, face_1_area)

                    for point in new_face_1 + [new_face_1[0]]:
                        current_point = self.get_point(point)
                        print(
                            current_point[0],
                            current_point[1],
                            current_point[2],
                            file=polygon_out,
                        )

                    print("\n", file=polygon_out)

                    new_face_index = self.add_face(new_face_2)

                    if face_index in self.fracture_faces:
                        self.fracture_faces.add(new_face_index)

                    for key in self.fracture_faces_multi:
                        if face_index in self.fracture_faces_multi[key]:
                            self.fracture_faces_multi[key].add(new_face_index)

                    if face_index in self.face_to_fracture:
                        self.face_to_fracture[new_face_index] = self.face_to_fracture[
                            face_index
                        ]
                        self.fracture_to_face[self.face_to_fracture[face_index]].append(
                            new_face_index
                        )

                    (face_area, face_centroid) = self.find_face_centroid(new_face_index)
                    self.set_face_real_centroid(new_face_index, face_centroid)
                    self.set_face_area(new_face_index, face_area)

                    self.set_face_normal(
                        new_face_index, self.get_face_normal(face_index)
                    )

                    done_faces.append(new_face_index)
                    faces = self.get_cell(cell_index)
                    self.set_cell_faces(cell_index, list(faces) + [new_face_index])
                    cell_orientations = self.get_cell_normal_orientation(cell_index)
                    local_face_index = list(self.get_cell(cell_index)).index(face_index)

                    if self.is_boundary_face(face_index, self.get_boundary_markers()):
                        boundary_marker = self.find_boundary_marker(
                            face_index, self.get_boundary_markers()
                        )
                        self.add_boundary_face(
                            boundary_marker,
                            new_face_index,
                            cell_orientations[local_face_index],
                        )

                    self.set_cell_orientation(
                        cell_index,
                        np.array(
                            list(cell_orientations)
                            + [cell_orientations[local_face_index]]
                        ),
                    )

                    cell_next_door = self.get_face_to_cell(face_index)
                    cell_next_door.remove(cell_index)

                    if len(cell_next_door) == 1:
                        next_door_faces = self.get_cell(cell_next_door[0])
                        next_door_local_face_index = list(next_door_faces).index(
                            face_index
                        )
                        next_door_faces = list(next_door_faces) + [new_face_index]
                        next_door_orientations = self.get_cell_normal_orientation(
                            cell_next_door[0]
                        )
                        next_door_orientations = list(next_door_orientations) + [
                            next_door_orientations[next_door_local_face_index]
                        ]
                        next_door_orientations = np.array(next_door_orientations)

                        self.set_cell_faces(cell_next_door[0], next_door_faces)
                        self.set_cell_orientation(
                            cell_next_door[0], next_door_orientations
                        )

                        if cell_next_door[0] in face_segments_to_be_added:
                            face_segments_to_be_added[cell_next_door[0]] += [
                                interior_face_segments[-1]
                            ]
                        else:
                            face_segments_to_be_added[cell_next_door[0]] = [
                                interior_face_segments[-1]
                            ]

                    for point in new_face_2 + [new_face_2[0]]:
                        current_point = self.get_point(point)
                        print(
                            current_point[0],
                            current_point[1],
                            current_point[2],
                            file=polygon_out,
                        )

                    print("\n", file=polygon_out)

            if cell_index in face_segments_to_be_added:
                interior_face_segments += face_segments_to_be_added[cell_index]

            ## Add the interior slice polygon.
            if len(interior_face_segments) > 0:
                try:
                    new_face = self.construct_polygon_from_segments(
                        interior_face_segments
                    )
                except:
                    print(cell_index)
                    1 / 0
                ## Divide the face in case the fracture is only partially
                ## penetrating.
                points_face_1 = []
                points_face_2 = []

                stage = 1
                for i in range(1):
                    v1 = self.get_point(new_face[i + 1]) - self.get_point(new_face[i])
                    v2 = self.get_point(new_face[i]) - self.get_point(new_face[i - 1])
                    new_face_normal = np.cross(v2, v1)

                new_face_normal /= np.linalg.norm(new_face_normal)

                if np.dot(new_face_normal, fracture.normal) > 0.0:
                    same_relative_rotation = True
                else:
                    same_relative_rotation = False

                has_intersection = False
                (
                    has_intersection,
                    poly_1_list,
                    poly2,
                ) = self.get_fracture_intersection_diff(fracture, new_face)

                new_face = [new_face]

                if has_intersection:
                    new_face = []
                    for poly in poly_1_list:
                        new_face.append([])
                        for point in poly:
                            print(point[0], point[1], point[2], file=clipping_out)
                            new_point_index = self.add_point(point)
                            new_face[-1].append(new_point_index)
                        print("\n", file=clipping_out)

                    new_faces_2_list = []

                    for poly in poly2:
                        new_face_2 = []
                        for point in poly:
                            print(point[0], point[1], point[2], file=clipping_2_out)
                            new_point_index = self.add_point(point)
                            new_face_2.append(new_point_index)
                        print("\n", file=clipping_2_out)

                        new_face_index_2 = self.add_face(new_face_2)
                        new_faces_2_list.append(new_face_index_2)
                        done_faces.append(new_face_index_2)
                        (face_area, face_centroid) = self.find_face_centroid(
                            new_face_index_2
                        )

                        self.set_face_real_centroid(new_face_index_2, face_centroid)
                        self.set_face_area(new_face_index_2, face_area)

                        new_face_normal = self.find_face_normal(new_face_index_2)
                        self.set_face_normal(new_face_index_2, new_face_normal)

                new_face_index_list = []

                for face in new_face:
                    new_face_index_list.append(self.add_face(face))

                for new_face_index in new_face_index_list:
                    if cell_index in cells_on_fracture:
                        self.fracture_faces.add(new_face_index)
                        fracture_faces.add(new_face_index)
                        self.fracture_faces_multi[current_key].add(new_face_index)

                        self.face_to_fracture[new_face_index] = fracture.id
                        self.fracture_to_face[fracture.id].append(new_face_index)

                    done_faces.append(new_face_index)

                    (face_area, face_centroid) = self.find_face_centroid(new_face_index)
                    self.set_face_real_centroid(new_face_index, face_centroid)
                    self.set_face_area(new_face_index, face_area)

                    new_face_normal = self.find_face_normal(new_face_index)
                    self.set_face_normal(new_face_index, new_face_normal)

                cell_faces = self.get_cell(cell_index)

                faces_for_cell_1 = []
                faces_for_cell_2 = []

                normals_for_cell_1 = []
                normals_for_cell_2 = []

                for face_index in self.get_cell(cell_index):
                    current_center = self.get_face_real_centroid(face_index)
                    plane_to_center = point_on_plane - current_center

                    if np.dot(plane_to_center, plane_normal) > 0.0:
                        faces_for_cell_1.append(face_index)
                        local_face_index = list(self.get_cell(cell_index)).index(
                            face_index
                        )
                        face_normal = self.get_cell_normal_orientation(cell_index)[
                            local_face_index
                        ]
                        normals_for_cell_1.append(face_normal)
                    else:
                        faces_for_cell_2.append(face_index)
                        local_face_index = list(self.get_cell(cell_index)).index(
                            face_index
                        )
                        face_normal = self.get_cell_normal_orientation(cell_index)[
                            local_face_index
                        ]
                        normals_for_cell_2.append(face_normal)
                ## Loops through the points for the intersecting face.
                ## Check that intersection happeend.
                for point_index in self.get_face(new_face_index):
                    self.add_point_to_other_faces(
                        faces_for_cell_1, self.get_point(point_index)
                    )
                    self.add_point_to_other_faces(
                        faces_for_cell_2, self.get_point(point_index)
                    )

                ## new_face_index should be changed to new_face_index_list
                ## only works because thef fracture face is a single face.
                ## new_face_index is coming from the for-loop a little earlier.
                faces_for_cell_1.append(new_face_index)
                faces_for_cell_2.append(new_face_index)

                if np.dot(new_face_normal, plane_normal) > 0.0:
                    normals_for_cell_1.append(1)
                    normals_for_cell_2.append(-1)
                else:
                    normals_for_cell_1.append(-1)
                    normals_for_cell_2.append(1)

                if has_intersection:
                    faces_for_cell_1 += new_faces_2_list
                    faces_for_cell_2 += new_faces_2_list

                    if np.dot(new_face_normal, plane_normal) > 0.0:
                        normals_for_cell_1 += [1] * len(new_faces_2_list)
                        normals_for_cell_2 += [-1] * len(new_faces_2_list)
                    else:
                        normals_for_cell_1 += [-1] * len(new_faces_2_list)
                        normals_for_cell_2 += [1] * len(new_faces_2_list)

                self.set_cell_faces(cell_index, faces_for_cell_1)
                self.set_cell_orientation(cell_index, normals_for_cell_1)

                (cell_volume, cell_centroid) = self.find_volume_centroid(cell_index)
                self.set_cell_real_centroid(cell_index, cell_centroid)
                self.set_cell_volume(cell_index, cell_volume)

                for face_index in faces_for_cell_2:
                    if has_intersection:
                        if (
                            face_index != new_face_index
                            and face_index not in new_faces_2_list
                        ):
                            self.remove_from_face_to_cell(face_index, cell_index)

                    elif face_index != new_face_index:
                        self.remove_from_face_to_cell(face_index, cell_index)

                    try:
                        pass
                    except:
                        print(
                            cell_index,
                            self.get_face_to_cell(face_index),
                            new_face_index_2,
                            face_index,
                        )
                        1 / 0

                new_cell_index = self.add_cell(faces_for_cell_2, normals_for_cell_2)

                self.intersecting_cells.append(1)

                (cell_volume, cell_centroid) = self.find_volume_centroid(new_cell_index)

                self.set_cell_volume(new_cell_index, cell_volume)
                self.set_cell_real_centroid(new_cell_index, cell_centroid)

                self.set_cell_k(new_cell_index, self.get_cell_k(cell_index))

        return fracture_faces
