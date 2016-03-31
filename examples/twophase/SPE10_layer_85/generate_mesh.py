import mimpy.mesh.hexmesh as hexmesh
import mimpy.mfd.mfd as mfd
import mimpy.models.twophase.twophase as twophase
import numpy as np


res_mfd = mfd.MFD()
res_mfd.set_m_e_construction_method(0)

#Define the permeability function
def res_k(p, i, j, k):
    return 1.e-10*np.eye(3)

def frac_k(p, i, j, k):
    return 1.e-6*np.eye(3)

#The modification function is applied to the points of the mesh. 
#In this case no change is applied. 
def mod_function(p, i, j, k):
    return p

def krw(se):
    if se<0:
        return 0.
    elif se>1.:
        return 1.
    return se**2

def kro(se):
    if se>1.:
        return 0.
    elif se<0.:
        return 1.
    return (1.-se)**2

res_mesh =  hexmesh.HexMesh()

permfile = open("spe_perm_layer85.dat")

res_mesh.build_mesh(51, 2, 61, res_k, 670., .6, 365., mod_function)

Kx= []
Ky = []
Kz = []

for cell_index in range(res_mesh.get_number_of_cells()):
    line = permfile.readline()
    line = line.split()
    line = map(float, line)
    Kx.append(line[0])
    Ky.append(line[1])
    Kz.append(line[2])

    current_k = line[0]*np.eye(3) 
    current_k *= 1.e-12
    res_mesh.set_cell_k((cell_index%60)*50+cell_index/60, current_k)

res_mesh.save_mesh(open("spe_10.mms", 'w'))

