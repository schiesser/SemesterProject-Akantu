import numpy as np
import akantu as aka
from variationnal_operator_function import *
from plot import *

print(aka.__file__)
print(aka.__version__)

# Mesh generation

mesh_file = """
Point(1) = {0, 0, 0, 0.5};
Point(2) = {1, 0, 0, 0.5};
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

# Interpolation

## create a field  
nodes = mesh.getNodes()

Ngroup = ShapeField(Sup)
Ngroup.evalOnQuadraturePoints()
print(Ngroup.value_integration_points)
Bgroup = GradientOperator(Ngroup)
Bgroup.evalOnQuadraturePoints()
print(Bgroup.value_integration_points)
intN = FieldIntegrator.integrate(Ngroup)
print(intN)
intB = FieldIntegrator.integrate(Bgroup)
print(intB)
K = Bgroup^Bgroup
K.evalOnQuadraturePoints()
print(K.value_integration_points)
intK = FieldIntegrator.integrate(K)
print(intK)
print(intK.reshape(3,3))
