import numpy as np
import akantu as aka
from variationnal_operator_function import *
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

#plotMesht3(nodes, conn,name_file="MeshTestTriangle3.png")#save the mesh in .png

##Support declaration

model = aka.SolidMechanicsModel(mesh)
model.initFull(_analysis_method=aka._static)

elem_filter = np.array([[0]])
fem = model.getFEEngine()
elem_type = aka._triangle_3
Sup = Support(elem_filter, fem, spatial_dimension, elem_type)
############################################
# Test :

## field dimension :
field_dimension = spatial_dimension

## tolerance :
tol = 10e-8

## array containing shape functions :
Ngroup = N(Sup,field_dimension)
resNgroup = Ngroup.evalOnQuadraturePoints()
print("N grouped :")
print(resNgroup)
print("avec shape :")
print(resNgroup.shape)

## Integration of N :
intN = FieldIntegrator.integrate(Ngroup)
print("integration de N :")
print(intN)
print("avec shape :")
print(intN.shape)

## Assembly of int(N) :
AssembledIntN=Assembly.assemblyV(intN, Sup, field_dimension)
print("assembly of int(N) :")
print(AssembledIntN)
# True result :
expected_result_integration_N = np.array([[1/12, 0, 1/12, 0, 1/12, 0, 1/12, 0, 1/6, 0],[0, 1/12, 0, 1/12, 0, 1/12, 0, 1/12, 0, 1/6]])
# control of the computed integration of N :
np.testing.assert_allclose(AssembledIntN, expected_result_integration_N, atol=tol, err_msg="integration of N isn't correct")


## Grad(N) :
Bgroup = GradientOperator(Ngroup)
resBgroup = Bgroup.evalOnQuadraturePoints()
print("B grouped :")
print(resBgroup)
print("with shape :")
print(resBgroup.shape)

## Integration of grad(N) :
intB = FieldIntegrator.integrate(Bgroup)
print("B integration :")
print(intB)
print("with shape :")
print(intB.shape)

## Assembly integrate grad(N):
AssembledIntB=Assembly.assemblyB(intB, Sup, field_dimension)
print("Assembly of int[grad(N)]")
print(AssembledIntB)
print(AssembledIntB.shape)
# True result :
expected_result_integration_gradN = expected_result = np.array([[-0.5, 0, 0.5, 0, 0.5, 0, -0.5, 0, 0, 0],[0, -0.25, 0, -0.25, 0, 0.25, 0, 0.25, 0, 0],[-0.25, -0.5, -0.25, 0.5, 0.25, 0.5, 0.25, -0.5, 0, 0]])
# control of the computed integration of grad(N) :
np.testing.assert_allclose(AssembledIntB, expected_result_integration_gradN, atol=tol, err_msg="integration of grad(N) isn't correct")


## Test operation Transpose(B)@B :
BtB = transpose(Bgroup)@Bgroup
resBtB = BtB.evalOnQuadraturePoints()
print("results BtB:")
print(resBtB)
print("with shape")
print(resBtB.shape)

## Integrate BtB :
intBtB = FieldIntegrator.integrate(BtB)
print("result of int[BtB] :")
print(intBtB)
print("avec shape :")
print(intBtB.shape)

## Assembly of K gloable :

Kglobale = Assembly.assemblyK(intBtB, Sup, field_dimension)
print("K globale :")
print(Kglobale)
print("with shape :")
print(Kglobale.shape)
# True result :
expected_result_K = np.array([[ 1.25,  0.5,  -0.375, -0.25,   0.0,   0.0,   0.375,  0.25,  -1.25, -0.5],[ 0.5,   1.25,  0.25,  -0.375,  0.0,   0.0,  -0.25,   0.375, -0.5,  -1.25],[-0.375, 0.25,  1.25,  -0.5,    0.375, -0.25,  0.0,    0.0,  -1.25,  0.5],[-0.25, -0.375, -0.5,   1.25,   0.25,  0.375,  0.0,    0.0,   0.5,  -1.25],[ 0.0,   0.0,   0.375,  0.25,   1.25,  0.5,   -0.375, -0.25, -1.25, -0.5],[ 0.0,   0.0,  -0.25,   0.375,  0.5,   1.25,   0.25,  -0.375, -0.5,  -1.25],[ 0.375, -0.25, 0.0,    0.0,   -0.375, 0.25,   1.25,  -0.5,  -1.25,  0.5],[ 0.25,  0.375, 0.0,    0.0,   -0.25, -0.375, -0.5,    1.25,  0.5,  -1.25],[-1.25, -0.5,  -1.25,   0.5,   -1.25, -0.5,   -1.25,   0.5,   5.0,    0.0],[-0.5,  -1.25,  0.5,   -1.25,  -0.5,  -1.25,  0.5,-1.25,  0.0, 5.0]])
# control of the computed stiffness global matrix :
np.testing.assert_allclose(Kglobale, expected_result_K, atol=tol, err_msg="Gloable Stiffness matrix isn't correct")