from assembly import *
import numpy as np
import akantu as aka
from variationnal_operator_function import *
import matplotlib.tri as tri
import matplotlib.pyplot as plt

print(aka.__file__)
print(aka.__version__)

## Mesh generation

mesh_file = """
Point(1) = {0, 0, 0, 1};
Point(2) = {1, 0, 0, 1};
Point(3) = {1, 1, 0, 1};
Point(4) = {0, 1, 0, 1};
"""
mesh_file += """
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
"""
mesh_file += """
Line Loop(5) = {1, 2, 3, 4};
"""
mesh_file += """
Plane Surface(6) = {5};
"""

open("triangle.geo", 'w').write(mesh_file)

# reading the mesh
spatial_dimension = 2
mesh_file = 'triangle.msh'
mesh = aka.Mesh(spatial_dimension)
mesh.read(mesh_file)

conn = mesh.getConnectivity(aka._triangle_3)
nodes = mesh.getNodes()
triangles = tri.Triangulation(nodes[:, 0], nodes[:, 1], conn)
t=plt.triplot(triangles, '--', lw=.8)
plt.savefig('MeshElementTriangle.png')

##Support declaration

model = aka.SolidMechanicsModel(mesh)
model.initFull(_analysis_method=aka._static)

elem_filter = np.array([[0]])
fem = model.getFEEngine()
elem_type = aka._triangle_3
ghost_type = aka.GhostType(1) #peu importe pour le moment
Sup = Support(elem_filter, fem, spatial_dimension, elem_type, ghost_type)
##########################
Ngroup = N(Sup,2)
resNgroup = Ngroup.evalOnQuadraturePoints()
print("N grouped :")
print(resNgroup)
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
"""
intB = FieldIntegrator2.integrate(Bgroup)
print("B integration :")
print(intB)
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
outputdim1 = 10 #nb_nodes *spatial_dim
outputdim2 = 10 #nb_nodes *spatial_dim

Kglobale = assemblyK(conn, K, outputdim1,outputdim2, 2)

print(Kglobale)