import numpy as np
import akantu as aka
from variationnal_operator_function import *
import matplotlib.pyplot as plt
from plot import *

print(aka.__file__)
print(aka.__version__)

## Mesh generation

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
plotMesht3(nodes, conn)

##Support declaration

model = aka.SolidMechanicsModel(mesh)
model.initFull(_analysis_method=aka._static)

elem_filter = np.array([[0]])
fem = model.getFEEngine()
elem_type = aka._triangle_3
ghost_type = aka.GhostType(1) #peu importe pour le moment
Sup = Support(elem_filter, fem, spatial_dimension, elem_type, ghost_type)
######################################################################
field_dim = 1
shapef = ShapeField(Sup, field_dim)
gradient = GradientOperator(shapef)

# K :
res_int=FieldIntegrator.integrate(transpose(gradient)@gradient)
K = Assembly.assemblyK(res_int,Sup,1)

# for boundary conditions :
tol =10e-6

index = np.arange(0,nodes.shape[0])
x=np.zeros(index.shape)

nodes_t0 = index[nodes[:,0]<tol]
nodes_t1 = index[nodes[:,0]>1-tol]

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

plt.scatter(nodes[:,0],nodes[:,1], c=x, cmap='viridis', s=40)
plt.colorbar(label='Temperature')

plt.title('Temperature value at each node')
plt.savefig("chaleur2D.png")