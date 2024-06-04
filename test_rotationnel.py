import numpy as np
import akantu as aka
from variationnal_operator_function import *
import matplotlib.pyplot as plt
from plot import *

print(aka.__file__)
print(aka.__version__)

## 1) Mesh generation

mesh_file = """
Point(1) = {0, 0, 0, 0.25};
Point(2) = {1, 0, 0, 0.25};
Point(3) = {1, 1, 0, 0.25};
Point(4) = {0, 1, 0, 0.25};
Point(5) = {0, 0, 10, 0.25};
Point(6) = {1, 0, 10, 0.25};
Point(7) = {1, 1, 10, 0.25};
Point(8) = {0, 1, 10, 0.25};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

Line Loop(13) = {1, 2, 3, 4};
Line Loop(14) = {5, 6, 7, 8};
Line Loop(15) = {1, 10, -5, -9};
Line Loop(16) = {2, 11, -6, -10};
Line Loop(17) = {3, 12, -7, -11};
Line Loop(18) = {4, 9, -8, -12};

Plane Surface(19) = {13};
Plane Surface(20) = {14};
Plane Surface(21) = {15};
Plane Surface(22) = {16};
Plane Surface(23) = {17};
Plane Surface(24) = {18};

Surface Loop(25) = {19, 20, 21, 22, 23, 24};
Volume(26) = {25};

Physical Surface("Surface1") = {19};
Physical Surface("Surface2") = {20};
Physical Surface("Surface3") = {21};
Physical Surface("Surface4") = {22};
Physical Surface("Surface5") = {23};
Physical Surface("Surface6") = {24};
Physical Volume("Volume") = {26};
"""

open("barre_fine.geo", 'w').write(mesh_file)
#.msh
nodes, conn = meshGeo("barre_fine.geo", dim =3, order=1, element_type='tetra')
# reading the mesh
spatial_dimension = 3
mesh_file = 'barre_fine.msh'
mesh = aka.Mesh(spatial_dimension)
mesh.read(mesh_file)
nodes = mesh.getNodes()
conn = mesh.getConnectivity(aka._tetrahedron_4)

#plotMeshtetra(nodes, conn)#save the mesh in .png

## 2) Support declaration

model = aka.SolidMechanicsModel(mesh)
model.initFull(_analysis_method=aka._static)

elem_filter = np.array([[0]])
fem = model.getFEEngine()
elem_type = aka._tetrahedron_4
ghost_type = aka.GhostType(1) #peu importe pour le moment
Sup = Support(elem_filter, fem, spatial_dimension, elem_type)
######################################################################
field_dim = 3

## 3) + 4) Write weak form (using differential operator) and integrate
shapef = ShapeField(Sup, field_dim)
rot = RotationalOperator(shapef)

res_int=FieldIntegrator.integrate(transpose(rot)@rot)

## 5) Assembly
K = Assembly.assemblyK(res_int,Sup,3)

## 6) Apply boundary conditions to solve the problem
# for boundary conditions :
tol =10e-6

index = np.arange(0,nodes.shape[0])
x=np.zeros(index.shape[0]*3)
mask_a0x = (nodes[:, 2] < tol) #select all nodes on plane z = 0
nodes_a0x = index[mask_a0x] * spatial_dimension
nodes_a0y = nodes_a0x+1
nodes_a0z = nodes_a0y+1

index_tot = np.concatenate((index*spatial_dimension, (index*spatial_dimension+1), (index*spatial_dimension+2)))
index_remove0 = np.concatenate((nodes_a0x,nodes_a0y,nodes_a0z)) #indice of component wiht boundary condition
index_to_keep = np.setdiff1d(index_tot,index_remove0) #indice of ddl

a0 = 20
x[index_remove0]=a0 #apply 20 as boundary condition

comp_t0 = np.sum(K[:,index_remove0], axis = 1)*a0
b = -comp_t0
b_f = b[index_to_keep]
A = K[:,index_to_keep]
A = A[index_to_keep,:]
x[index_to_keep] = np.linalg.solve(A, b_f)
x=x.reshape(nodes.shape) #array x contains displacement, same shape as the nodes array (which contain the coordinates of the nodes)

# save results in CSV file
#import pandas as pd
#df = pd.DataFrame({'x': nodes[:, 0], 'y': nodes[:, 1],'z': nodes[:, 2], 'vx': x[:, 0],'vy': x[:, 1], 'vz': x[:, 2]})
#df.to_csv('barre_paraview.csv', index=False)