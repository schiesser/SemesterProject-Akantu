import numpy as np
import akantu as aka
from variationnal_operator_function import *
from plot import *

print(aka.__file__)
print(aka.__version__)

## Mesh generation 

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
#.msh
points, conn = meshGeo('segment.geo', dim=1, order=1)
# reading the mesh
spatial_dimension = 1
mesh_file = 'segment.msh'
mesh = aka.Mesh(spatial_dimension)
mesh.read(mesh_file)
#plotMeshs(points, conn, name_file="MeshTestSegment2.png" ) #save the mesh in .png

## Support declaration
model = aka.SolidMechanicsModel(mesh)
model.initFull(_analysis_method=aka._static)
elem_filter = np.array([[0]])
fem = model.getFEEngine()
elem_type = aka._segment_2
Sup = Support(elem_filter, fem, spatial_dimension, elem_type)
######################################################################
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
print("with shape :")
print(resNgroup.shape)

## Integration of N :
intN = FieldIntegrator.integrate(Ngroup)
print("integrationof N :")
print(intN)
print("with shape :")
print(intN.shape)

## Assembly of int(N) :
AssembledIntN=Assembly.assemblyV(intN, Sup, field_dimension)
print("int(N) assembled :")
print(AssembledIntN)
# True result :
expected_result_integration_N = np.array([[0.125,0.5,0.375]])
# control of the computed integration of N :
np.testing.assert_allclose(AssembledIntN, expected_result_integration_N, atol=tol, err_msg="integration of N isn't correct")

## Grad(N) :
Bgroup = GradientOperator(Ngroup)
resBgroup = Bgroup.evalOnQuadraturePoints()
print("B grouped :")
print(resBgroup)
print("with shape :")
print(resBgroup.shape)

## Integrate grad(N) :
intB = FieldIntegrator.integrate(Bgroup)
print("B integration :")
print(intB)
print("with shape :")
print(intB.shape)

## Assembly of integrate of grad(N):
AssembledIntB=Assembly.assemblyV(intB, Sup, field_dimension)
print("Assembly of int[grad(N)]")
print(AssembledIntB)
# True result :
expected_result_integration_gradN = np.array([[-1.0,0.0,1.0]])
# control of the computed integration of grad(N) :
np.testing.assert_allclose(AssembledIntB, expected_result_integration_gradN, atol=tol, err_msg="integration of grad(N) isn't correct")

## Test operation Transpose(B)@B :
BtB = transpose(Bgroup)@Bgroup
resBtB = BtB.evalOnQuadraturePoints()
print("result of BtB:")
print(resBtB)
print("with shape")
print(resBtB.shape)

## Integrate BtB :
intBtB = FieldIntegrator.integrate(BtB)
print("result of int[BtB] :")
print(intBtB)
print("with shape :")
print(intBtB.shape)

## Assembly  K gloable :

Kglobale = Assembly.assemblyK(intBtB, Sup, field_dimension)
print("K globale :")
print(Kglobale)
print("with shape :")
print(Kglobale.shape)
# True result :
expected_result_K = np.array([[4.0,-4.0,0.0],[-4.0,16/3,-4/3],[0.0,-4/3,4/3]])
# control of the computed stiffness global matrix :
np.testing.assert_allclose(Kglobale, expected_result_K, atol=tol, err_msg="Gloable Stiffness matrix isn't correct")
