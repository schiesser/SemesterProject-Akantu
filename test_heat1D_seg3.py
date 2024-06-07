import os
import subprocess
import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import akantu as aka
from variationnal_operator_function import *
from plot import *

## 1) Mesh generation

mesh_file = """
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
Line(1) = {1, 2};
"""
open("segment3.geo", 'w').write(mesh_file)
#.msh
points, conn = meshGeo('segment3.geo', dim=1, order=2, element_type='line3')
# Reading the mesh
spatial_dimension = 1    
mesh_file = 'segment3.msh'
mesh = aka.Mesh(spatial_dimension)
mesh.read(mesh_file)

#plotMeshs2(points, conn)#save the mesh in .png

## 2) Support declaration

model = aka.SolidMechanicsModel(mesh)
model.initFull(_analysis_method=aka._static)

elem_filter = np.array([[0]])
fem = model.getFEEngine()
elem_type = aka._segment_3
Sup = Support(elem_filter, fem, spatial_dimension, elem_type)
######################################################################
field_dim = 1

## 3) + 4) Write weak form (using differential operator) and integrate
t = ShapeField(Sup,field_dim)

res_int=FieldIntegrator.integrate(transpose(Grad(t))@Grad(t))

## 5) Assembly
K = Assembly.assemblyK(res_int,Sup,field_dim)

## 6) Apply boundary conditions to solve the problem
tol =10e-6

index = np.arange(0,points.shape[0])
x=np.zeros(index.shape)
nodes_t0 = index[points[:,0]<tol]#select indice at x=0
nodes_t1 = index[points[:,0]>1-tol]#select indice at x=1

index_remove = np.concatenate((nodes_t0, nodes_t1))
index_to_keep = np.setdiff1d(index, index_remove) #ddl

t0 = 20
t1 = 10
x[nodes_t0]=t0
x[nodes_t1]=t1
comp_t0 = np.sum(K[:,nodes_t0], axis = 1)*t0
comp_t1 = np.sum(K[:,nodes_t1], axis = 1)*t1
b = -comp_t0-comp_t1
b_f = b[index_to_keep]
A = K[:,index_to_keep]
A = A[index_to_keep,:]

x[index_to_keep] = np.linalg.solve(A, b_f)

# Plot results
plotMeshs(points, conn, nodal_field=x, title ='Temperature value',name_file = "chaleur1Ds3.png" )
