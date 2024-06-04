import numpy as np
import akantu as aka
from variationnal_operator_function import *
from plot import *

# Mesh generation 
mesh_file = """
Point(1) = {0, 0, 0, 0.5};
Point(2) = {1, 0, 0, 0.5};

Line(1) = {1, 2};
"""

open("segment3.geo", 'w').write(mesh_file)
#.msh
points, conn = meshGeo('segment3.geo', dim=1, order=2, element_type='line3')
# reading the mesh
spatial_dimension = 1    
mesh_file = 'segment3.msh'
mesh = aka.Mesh(spatial_dimension)
mesh.read(mesh_file)

#plotMeshs(points, conn,name_file="MeshTestSegment3.png")#save the mesh in .png

##Support declaration

model = aka.SolidMechanicsModel(mesh)
model.initFull(_analysis_method=aka._static)

elem_filter = np.array([[0]])
fem = model.getFEEngine()
elem_type = aka._segment_3
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
print("integration of N :")
print(intN)
print("avec shape :")
print(intN.shape)

## Assembly of int(N) :
AssembledIntN=Assembly.assemblyV(intN, Sup, field_dimension)
print("assembly of int(N) :")
print(AssembledIntN)
# True result :
expected_result_integration_N = np.array([[1/12, 1/12, 1/6, 1/3, 1/3]])
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
AssembledIntB=Assembly.assemblyV(intB, Sup, field_dimension)
print("Assembly int[grad(N)]")
print(AssembledIntB)
# True result :
expected_result_integration_gradN = np.array([[-1.0, 1.0, 0, 0, 0]])
# control of the computed integration of grad(N) :
np.testing.assert_allclose(AssembledIntB, expected_result_integration_gradN, atol=tol, err_msg="integration of grad(N) isn't correct")


## Test operation Transpose(B)@B :
BtB = transpose(Bgroup)@Bgroup
resBtB = BtB.evalOnQuadraturePoints()
print("resultat BtB:")
print(resBtB)
print("with shape")
print(resBtB.shape)

## Integration of BtB :
intBtB = FieldIntegrator.integrate(BtB)
print("result integration of BtB :")
print(intBtB)
print("with shape :")
print(intBtB.shape)

## Assembly K gloable :

Kglobale = Assembly.assemblyK(intBtB, Sup, field_dimension)
print("K globale :")
print(Kglobale)
print("with shape :")
print(Kglobale.shape)
# True result :
expected_result_K = np.array([[ 14/3, 0.0, 2/3, -16/3, 0.0],[0.0, 14/3, 2/3, 0.0, -16/3],[2/3, 2/3, 28/3, -16/3, -16/3],[-16/3, 0.0, -16/3, 32/3, 0.0],[0.0, -16/3, -16/3, 0.0, 32/3]])
# control of the computed stiffness global matrix :
np.testing.assert_allclose(Kglobale, expected_result_K, atol=tol, err_msg="Gloable Stiffness matrix isn't correct")