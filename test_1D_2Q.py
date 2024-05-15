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
plotMeshs3(points, conn)

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

## array contenant les N :
field_dimension = 1
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
