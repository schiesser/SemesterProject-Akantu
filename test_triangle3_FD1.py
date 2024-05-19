import numpy as np
import akantu as aka
from variationnal_operator_function import *
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from plot import *

print(aka.__file__)
print(aka.__version__)

## Mesh generation

mesh_file = """
Point(1) = {0, 0, 0, 1};
Point(2) = {0.5, 0, 0, 1};
Point(3) = {0.5, 1, 0, 1};
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
Physical Surface("Mesh") = {6};
"""

open("triangle.geo", 'w').write(mesh_file)
#.msh
nodes, conn = meshGeo("triangle.geo", dim =2, order=1, element_type='triangle')
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

## field dimension :
field_dimension = 1

## tolerance :
tol = 10e-8

## array contenant les N :
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
AssembledIntN=Assembly.assemblyV(intN, Sup, field_dimension)
print("intégration de N assemblé :")
print(AssembledIntN)
# True result :
expected_result_integration_N = np.array([[1/12, 1/12, 1/12, 1/12, 1/6]])
# control of the computed integration of N :
error = np.abs(AssembledIntN-expected_result_integration_N)
assert error.all()<tol, "integration of N isn't correct"

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

## Assemblage de l'intégration de grad(N):
AssembledIntB=Assembly.assemblyB(intB, Sup, field_dimension)
print("Assemblage de l'intégration de grad(N)")
print(AssembledIntB)
print(AssembledIntB.shape)
# True result :
expected_result_integration_gradN = np.array([[-1/8, 1/8, 1/8, -1/8, 0],[3/8, 1/8, -3/8, -1/8, 0],[1/4, 1/4, 1/4, 1/4, 0]])
# control of the computed integration of grad(N) :
error = np.abs(AssembledIntB - expected_result_integration_gradN)
assert error.all()<tol, "integration of grad(N) isn't correct"

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

Kglobale = Assembly.assemblyK(intBtB, Sup, field_dimension)
print("K globale :")
print(Kglobale)
print("avec shape :")
print(Kglobale.shape)
# True result :
expected_result_K = np.array([[2.75, 0.75, 0.25, 0.125, 0, 0, -0.75, -1, -1, -1.25],[0.75, 2.75, 0, 0.25, 0, 0, -0.125, -0.75, -0.5, -1],[0.25, 0, 0.75, 0.25, -0.75, -0.125, 0, 0, 0.25, 0.25],[0.125, 0.25, 0.25, 0.75, -1, -0.75, 0, 0, -0.25, 0.25],[0, 0, -0.75, -1, 2.75, 0.75, 0.25, 0.125, -1, -1.25],[0, 0, -0.125, -0.75, 0.75, 2.75, 0, 0.25, -0.5, -1],[-0.75, -0.125, 0, 0, 0.25, 0, 0.75, 0.25, 0.25, 0.25],[-1, -0.75, 0, 0, 0.125, 0.25, 0.25, 0.75, -0.25, 0.25],[-1, -0.5, 0.25, -0.25, -1, -0.5, 0.25, -0.25, 3, 1],[-1.25, -1, 0.25, 0.25, -1.25, -1, 0.25, 0.25, 1, 3]])
# control of the computed integration of grad(N) :
error = np.abs(Kglobale-expected_result_K)
assert error.all()<tol, "Gloable Stiffness matrix isn't correct"