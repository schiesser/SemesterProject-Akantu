import os
import subprocess
import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import akantu as aka
from variationnal_operator_function import *
from plot import *

# Génération du maillage
mesh_file = """
Point(1) = {0, 0, 0, 0.5};
Point(2) = {1, 0, 0, 0.5};

Line(1) = {1, 2};
"""

open("segment3.geo", 'w').write(mesh_file)
points, conn = meshGeo('segment3.geo', dim=1, order=2, element_type='line3')
plotMeshs2(points, conn)

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
######################################################################
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
expected_result_integration_N = np.array([[1/12, 1/12, 1/6, 1/3, 1/3]])
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
expected_result_integration_gradN = np.array([[-1.0, 1.0, 0, 0, 0]])
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
expected_result_K = np.array([[ 14/3, 0.0, 2/3, -16/3, 0.0],[0.0, 14/3, 2/3, 0.0, -16/3],[2/3, 2/3, 28/3, -16/3, -16/3],[-16/3, 0.0, -16/3, 32/3, 0.0],[0.0, -16/3, -16/3, 0.0, 32/3]])
# control of the computed stiffness global matrix :
np.testing.assert_allclose(Kglobale, expected_result_K, atol=tol, err_msg="Gloable Stiffness matrix isn't correct")