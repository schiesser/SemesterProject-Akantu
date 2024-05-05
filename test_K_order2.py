import os
import subprocess
import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import akantu as aka
from variationnal_operator_function import *

def readMesh(filename, element_type='line3'):
    mesh = meshio.read(filename)
    return mesh.points[:, :2], mesh.cells_dict[element_type]

def meshGeo(filename, dim=2, order=2):  # Modifier la dimension à 2
    out = os.path.splitext(filename)[0] + '.msh'
    ret = subprocess.run(f"gmsh -2 -order {order} -o {out} {filename}", shell=True)
    if ret.returncode:
        print("Attention, gmsh n'a pas pu s'exécuter : le maillage n'est pas généré")
    else:
        print("Maillage généré")
        mesh = readMesh(out)
        return mesh
    return None

def plotMesh(points, conn):
    plt.figure(figsize=(8, 6))
    plt.plot(points[:, 0], points[:, 1], 'ko', markersize=3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Maillage')
    plt.axis('equal')
    plt.savefig('segment3mesh.png')

# Génération du maillage
mesh_file = """
Point(1) = {0, 0, 0, 0.5};
Point(2) = {1, 0, 0, 0.5};

Line(1) = {1, 2};
"""

open("segment3.geo", 'w').write(mesh_file)
points, conn = meshGeo('segment3.geo', dim=1, order=2)
plotMesh(points, conn)

# Lecture du maillage
spatial_dimension = 1    
mesh_file = 'segment3.msh'
mesh = aka.Mesh(spatial_dimension)
mesh.read(mesh_file)

model = aka.SolidMechanicsModel(mesh)
model.initFull(_analysis_method=aka._static)

elem_filter = np.array([[0]])
fem = model.getFEEngine()
elem_type = aka._segment_3
ghost_type = aka.GhostType(1) #peu importe pour le moment
Sup = Support(elem_filter, fem, spatial_dimension, elem_type, ghost_type)

Ngroup = ShapeField(Sup)
resNgroup = Ngroup.evalOnQuadraturePoints()
print("N grouped :")
print(resNgroup)

"""Bgroup = GradientOperator(Ngroup)
resBgroup = Bgroup.evalOnQuadraturePoints()
print("B grouped :")
print(resBgroup)
intN = FieldIntegrator.integrate(Ngroup)
print("N integration :")
print(intN)
"""