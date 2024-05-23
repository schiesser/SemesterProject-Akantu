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
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};

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
shapef = ShapeField(Sup)
gradient = GradientOperator(shapef)

# K :
res_int=FieldIntegrator.integrate(transpose(gradient)@gradient)
K = Assembly.assemblyK(res_int,Sup,1)

tol =10e-6

index = np.arange(0,points.shape[0])
x=np.zeros(index.shape)

nodes_t0 = index[points[:,0]<tol]
nodes_t1 = index[points[:,0]>1-tol]

index_remove = np.concatenate((nodes_t0, nodes_t1))
index_to_keep = np.setdiff1d(index, index_remove) #déjà dans le bonne ordre !

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

plt.scatter(points[:,0],points[:,1], c=x, cmap='viridis', s=40)
plt.colorbar(label='temperature')
plt.title('Temperature value at each node')
plt.savefig("chaleur1Dseg3.png")