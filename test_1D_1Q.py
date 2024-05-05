from assembly import *
import numpy as np
import akantu as aka
from variationnal_operator_function import *
from plot import *

print(aka.__file__)
print(aka.__version__)

# Mesh generation

mesh_file = """
Point(1) = {0, 0, 0, 0.5};
Point(2) = {1, 0, 0, 0.5};
"""
mesh_file += """
Line(1) = {1, 2};
"""
open("segment.geo", 'w').write(mesh_file)
points, conn = meshGeo('segment.geo', dim=1, order=1)
plotMesh(points, conn)

## reading the mesh
spatial_dimension = 1    
mesh_file = 'segment.msh'
mesh = aka.Mesh(spatial_dimension)
mesh.read(mesh_file)

# Support declaration

model = aka.SolidMechanicsModel(mesh)
model.initFull(_analysis_method=aka._static)

elem_filter = np.array([[0]])
fem = model.getFEEngine()
elem_type = aka._segment_2
ghost_type = aka.GhostType(1) #peu importe pour le moment
Sup = Support(elem_filter, fem, spatial_dimension, elem_type, ghost_type)

Ngroup = N(Sup,2)
resNgroup = Ngroup.evalOnQuadraturePoints()
print("N grouped :")
print(resNgroup.shape)
intN = FieldIntegrator2.integrate(Ngroup)
print("N integration :")
print(intN)
"""
assembledintN = FieldIntegrator2.assembly(Ngroup, intN)
print("N integration assembled :")
print(assembledintN)
"""
Bgroup = GradientOperator(Ngroup)
resBgroup = Bgroup.evalOnQuadraturePoints()
print("B grouped :")
print(resBgroup)
intB = FieldIntegrator2.integrate(Bgroup)
print("B integration :")
print(intB)
"""
assembledintB = FieldIntegrator2.assembly(Bgroup, intB)
print("B integration assembled :")
print(assembledintB)
"""

BtB = transpose(Bgroup)^Bgroup
test = BtB.evalOnQuadraturePoints()
print("shape output:")
print(test.shape)

K = FieldIntegrator2.integrate(BtB)
print(K)
outputdim1 = 3 #nb_nodes * spatial_dim
outputdim2 = 6 #nb_nodes * spatial_dim

Kglobale = assemblyK(conn, K, outputdim1,outputdim2, 1)

print(Kglobale)
print("test : ")
V=intN
Vglobal=assemblyV(conn, V,outputdim2, 2)
print(Vglobal)

V2=intB
V2global=assemblyV(conn, V2,outputdim2, 2)
print(V2global)