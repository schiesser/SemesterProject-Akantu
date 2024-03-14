import numpy as np
import akantu as aka
from variationnal_operator_function import *
from plot import *

## Mesh generation

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

# reading the mesh
spatial_dimension = 1    
mesh_file = 'segment.msh'
mesh = aka.Mesh(spatial_dimension)
mesh.read(mesh_file)

##Support declaration

model = aka.SolidMechanicsModel(mesh)
model.initFull(_analysis_method=aka._static)

elem_filter = np.array([[0]])
fem = model.getFEEngine()
spatial_dimension = 1
elemtype = aka._segment_2
ghost_type = aka.GhostType(1) #peu importe
Sup = Support(elem_filter, fem, spatial_dimension, elemtype, ghost_type)

## Interpolation

# create a field  
nodes = mesh.getNodes()
nodal_field=np.ones(nodes.shape)*2
nodal_field[0,0]=1
nodal_field[2,0]=0.5
print(nodal_field)

# output vector (will be filled by the method "interpolateOnIntegrationPoints")
## to adapt depending on the number of quadrature points (support)
nbr_elem = mesh.getConnectivity(Sup.elemtype).shape[0]
output=np.zeros((nbr_elem,1))

NTF = NodalTensorField("ex_displacement", Sup, nodal_field)

output = aka.ElementTypeMapArrayReal()
output.initialize(mesh, nb_component=1)
NTF.evalOnQuadraturePoints(output)

value_on_quadpoints=output(aka._segment_2)

print(value_on_quadpoints)

shapeDer=Sup.fem.getShapesDerivatives(Sup.elemtype)
print(shapeDer)
print(shapeDer.shape)
print(shapeDer[0,1])
gradu=np.array([2,-3])
print(value_on_quadpoints.shape)
gradu=gradu.reshape(value_on_quadpoints.shape)
print(gradu)

Field_int=FieldIntegrator()
print(Field_int.integrate(NTF, Sup, mesh))
