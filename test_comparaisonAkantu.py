import numpy as np
import akantu as aka
from variationnal_operator_function import *
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from plot import *

print(aka.__file__)
print(aka.__version__)

## Mesh generation
#.geo
mesh_file = """
Point(1) = {0, 0, 0, 1};
Point(2) = {1, 0, 0, 1};
Point(3) = {1, 1, 0, 1};
Point(4) = {0, 1, 0, 1};
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
Physical Line("BlockY") = {1};
Physical Line("BlockX") = {1};
Physical Line("Traction") = {3};
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

##Plot
conn = mesh.getConnectivity(aka._triangle_3)
nodes = mesh.getNodes()
triangles = tri.Triangulation(nodes[:, 0], nodes[:, 1], conn)
t=plt.triplot(triangles, '--', lw=.8)
plt.savefig('MeshElementTriangle.png')

## Material File
material_file = """
material elastic [
    name = steel
    rho = 7800     # density
    E   = 2.1e11   # young's modulus
    nu  = 0.3      # poisson's ratio
]"""
open('material.dat', 'w').write(material_file)
material_file = 'material.dat'
aka.parseInput(material_file)
E   = 2.1e11   # young's modulus
nu  = 0.3      # poisson's ratio

## Solid MechanicsModel
model = aka.SolidMechanicsModel(mesh)
model.initFull(_analysis_method=aka._static)

##Support declaration
elem_filter = np.array([[0]])
fem = model.getFEEngine()
elem_type = aka._triangle_3
ghost_type = aka.GhostType(1)
Sup = Support(elem_filter, fem, spatial_dimension, elem_type, ghost_type)
############################################
tol =10e-6

# Calcul depl. avec operateur diff. :

Ngroup = N(Sup, 2)
B = GradientOperator(Ngroup)

# D and reshape it to do the contraction
D = E/(1-nu**2)*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
number_elem = (Ngroup.nb_elem)
D = ConstitutiveLaw(number_elem, D)

BtDB = transpose(B)@D@B
K_locales = FieldIntegrator.integrate(BtDB)

K =Assembly.assemblyK(K_locales,Sup,2)

# K reducted
field_dim =2
index = np.arange(0,nodes.shape[0]*field_dim)
## nodes boundary condition
index_nodes_boundary_condition_x = index[::field_dim][nodes[:,1]<tol]
index_nodes_boundary_condition_y = index[1::field_dim][nodes[:,1]<tol]
index_remove = np.sort(np.concatenate((index_nodes_boundary_condition_x, index_nodes_boundary_condition_y)))
ddl = np.setdiff1d(index, index_remove)

K_reduced = K[:,ddl]
K_reduced = K_reduced[ddl,:]

# force 
f = np.zeros((K.shape[0]))
index_f_y = index[1::field_dim][nodes[:,1]>1-tol]
f[index_f_y]=5000
b = f[ddl]
print("nodes :")
print(nodes)
print("external force: ")
print(f.reshape(nodes.shape))

#linear system
x=np.zeros(index.shape)
x[ddl] = np.linalg.solve(K_reduced, b)
print("displacement :")
print(x.reshape(nodes.shape))

############################################
# Calcul deplacement avec Akantu

# set the displacement/Dirichlet boundary conditions
model.applyBC(aka.FixedValue(0.0, aka._x), "BlockX")
model.applyBC(aka.FixedValue(0.0, aka._y), "BlockY")

# set the force/Neumann boundary conditions
model.getExternalForce()[:] = 0.0

trac = [0.0, 10000] # Newtons/m^2

model.applyBC(aka.FromTraction(trac), "Traction")

blocked_dofs = model.getBlockedDOFs()
fext = model.getExternalForce()

print("blocked ddl :")
print(blocked_dofs)
print("external force")
print(fext)

# configure the linear algebra solver
solver = model.getNonLinearSolver()
solver.set("max_iterations", 2)
solver.set("threshold", 1e-8)
solver.set("convergence_type", aka.SolveConvergenceCriteria.residual)

# compute the solution
model.solveStep()

# extract the displacements
u = model.getDisplacement()
print(u)