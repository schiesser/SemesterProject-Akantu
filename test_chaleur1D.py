import numpy as np
import akantu as aka
from variationnal_operator_function import *
from plot import *

print(aka.__file__)
print(aka.__version__)

# Mesh generation

mesh_file = """
Point(1) = {0, 0, 0, 0.05};
Point(2) = {1, 0, 0, 0.05};
"""
mesh_file += """
Line(1) = {1, 2};
"""
open("segment.geo", 'w').write(mesh_file)
points, conn = meshGeo('segment.geo', dim=1, order=1)
plotMesh(points, conn)

## reading the mesh
spatial_dimension = 1    
mesh_file = 'segment.msh'
mesh = aka.Mesh(spatial_dimension)
mesh.read(mesh_file)

# Support declaration

model = aka.SolidMechanicsModel(mesh)
model.initFull(_analysis_method=aka._static)

elem_filter = np.array([[0]])
fem = model.getFEEngine()
elem_type = aka._segment_2
ghost_type = aka.GhostType(1) #peu importe pour le moment
Sup = Support(elem_filter, fem, spatial_dimension, elem_type, ghost_type)
######################################################################
shapef = ShapeField(Sup)
gradient = GradientOperator(shapef)

grad = gradient.evalOnQuadraturePoints()

dk = transpose(gradient)@gradient
toint=dk.evalOnQuadraturePoints()

res_int=FieldIntegrator.integrate(dk)

K = Assembly.assemblyK(res_int,Sup,1)

tol =10e-6

index = np.arange(0,points.shape[0])

nodes_t0 = index[points[:,0]<tol]
nodes_t1 = index[points[:,0]>1-tol]

index_remove = np.concatenate((nodes_t0, nodes_t1))
index_to_keep = np.setdiff1d(index, index_remove) #déjà dans le bonne ordre !

t0 = 20
t1 = 10

comp_t0 = np.sum(K[:,nodes_t0], axis = 1)*t0
comp_t1 = np.sum(K[:,nodes_t1], axis = 1)*t1
b = -comp_t0-comp_t1
b_f = b[index_to_keep]
A = K[:,index_to_keep]
A = A[index_to_keep,:]

x = np.linalg.solve(A, b_f)
print(x)
plt.scatter(points[:,0][index_to_keep],np.zeros_like(points[:,0][index_to_keep]), c=x, cmap='viridis', s=40)
plt.colorbar(label='temperature')
plt.title('Temperature value at each node')
plt.savefig("chaleur1D.png")