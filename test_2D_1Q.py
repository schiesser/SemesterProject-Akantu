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
############################################
# Début des tests :

## array contenant les N :
field_dimension = 2
Ngroup = N(Sup,field_dimension)
resNgroup = Ngroup.evalOnQuadraturePoints()
print("N grouped :")
print(resNgroup)
print("avec shape :")
print(resNgroup.shape)

## Integration de N :
intN = FieldIntegrator.integrate(Ngroup)
print("integration de N :")
print(intN)
print("avec shape :")
print(intN.shape)

## Assemblage de l'integration de N :
outputdim = 10 #nb_nodes * spatial_dim

AssembledIntN=Assembly.assemblyV(conn,intN,outputdim, field_dimension)
print("intégration de N assemblé :")
print(AssembledIntN)

## Gradient de N :
Bgroup = GradientOperator(Ngroup)
resBgroup = Bgroup.evalOnQuadraturePoints()
print("B grouped :")
print(resBgroup)
print("avec shape :")
print(resBgroup.shape)

## Integration de grad(N) :
intB = FieldIntegrator.integrate(Bgroup)
print("B integration :")
print(intB)
print("avec shape :")
print(intB.shape)

# !!! PROBLEM!!!
"""
## Assemblage de l'intégration de grad(N):
outputdim = 10 #nb_nodes * spatial_dim
AssembledIntB=Assembly.assemblyV(conn, intB,outputdim, field_dimension)
print("Assemblage de l'intégration de grad(N)")
print(AssembledIntB)
"""

## Test opération Transpose(B)@B :
BtB = transpose(Bgroup)@Bgroup
resBtB = BtB.evalOnQuadraturePoints()
print("resultat BtB:")
print(resBtB)
print("avec shape")
print(resBtB.shape)

## Intégration de BtB :
intBtB = FieldIntegrator.integrate(BtB)
print("résultat de l'intégration de BtB :")
print(intBtB)
print("avec shape :")
print(intBtB.shape)

## Assemblage de K gloable :
outputdim1 = 10 #nb_nodes * spatial_dim
outputdim2 = 10 #nb_nodes * spatial_dim

Kglobale = Assembly.assemblyK(conn, intBtB, outputdim1,outputdim2, field_dimension)
print("K globale :")
print(Kglobale)
print("avec shape :")
print(Kglobale.shape)