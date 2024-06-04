import numpy as np
import akantu as aka
from variationnal_operator_function import *
import matplotlib.pyplot as plt
from plot import *

print(aka.__file__)
print(aka.__version__)

## 1) Mesh generation

mesh_file = """
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
Point(3) = {1, 1, 0, 0.1};
Point(4) = {0, 1, 0, 0.1};
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

#plotMesht3(nodes, conn)#save the mesh in .png

## 2) Support declaration

model = aka.SolidMechanicsModel(mesh)
model.initFull(_analysis_method=aka._static)

elem_filter = np.array([[0]])
fem = model.getFEEngine()
elem_type = aka._triangle_3
Sup = Support(elem_filter, fem, spatial_dimension, elem_type)
######################################################################
field_dim = 1

## 3) + 4) Write weak form (using differential operator) and integrate
shapef = ShapeField(Sup, field_dim)
gradient = GradientOperator(shapef)

res_int=FieldIntegrator.integrate(transpose(gradient)@gradient)

## 5) Assembly
K = Assembly.assemblyK(res_int,Sup,1)

## 6) Apply boundary conditions to solve the problem
tol =10e-6

index = np.arange(0,nodes.shape[0])
x=np.zeros(index.shape)
nodes_t0 = index[nodes[:,0]<tol]#select indice at x=0
nodes_t1 = index[nodes[:,0]>1-tol]#select indice at x=1

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
plotMesht3(nodes, conn, nodal_field=x, title ='Temperature value',name_file = "chaleur2D.png" )