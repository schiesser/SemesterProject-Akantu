import numpy as np
import akantu as aka
from variationnal_operator_function import *
from plot import *

print(aka.__file__)
print(aka.__version__)

# Mesh generation

mesh_file = """
Point(1) = {0, 0, 0, 0.25};
Point(2) = {0.25, 0, 0, 0.75};
Point(3) = {1, 0, 0, 0.75};
"""
mesh_file += """
Line(1) = {1, 2};
Line(2) = {2, 3};
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
######################################################################
# Début des tests :

## field dimension :
field_dimension = 1

##tolerance :
tol = 10e-6

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
AssembledIntN=Assembly.assemblyV(intN, Sup, field_dimension)
print("intégration de N assemblé :")
print(AssembledIntN)
# True result :
expected_result_integration_N = np.array([[0.125,0.0,0.5,0.0,0.375,0.0],[0.0,0.125,0.0,0.5,0.0,0.375]])
# control of the computed integration of N :
np.testing.assert_allclose(AssembledIntN, expected_result_integration_N, atol=tol, err_msg="integration of N isn't correct")

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
AssembledIntB=Assembly.assemblyV(intB, Sup, field_dimension)
print("Assemblage de l'intégration de grad(N)")
print(AssembledIntB)
# True result :
expected_result_integration_gradN = np.array([[-1.0, 0.0,0.0,0.0,1.0,0.0],[0.0,-1.0,0.0,0.0,0.0,1.0]])
# control of the computed integration of grad(N) :
np.testing.assert_allclose(AssembledIntB, expected_result_integration_gradN, atol=tol, err_msg="integration of grad(N) isn't correct")


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
expected_result_K = np.array([[4.0,0.0,-4.0,0.0,0.0,0.0],[0.0,4.0,0.0,-4.0,0.0,0.0],[-4.0,0.0,16/3,0.0,-4/3,0.0],[0.0,-4.0,0.0,16/3,0.0,-4/3],[0.0,0.0,-4/3,0.0,4/3,0.0],[0.0,0.0,0.0,-4/3,0.0,4/3]])
# control of the computed stiffness global matrix :
np.testing.assert_allclose(Kglobale, expected_result_K, atol=tol, err_msg="Gloable Stiffness matrix isn't correct")
