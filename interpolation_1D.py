import numpy as np
import akantu as aka
from variationnal_operator_function import *
from plot import *

## Mesh generation

mesh_file = """
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
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

elem_filter = np.array([])
fem = model.getFEEngine()
spatial_dimension = 1
elemtype = aka._segment_2
ghost_type = aka.GhostType(1) #peu importe
Sup = Support(elem_filter, fem, spatial_dimension, elemtype, ghost_type)

## Interpolation

# create a field  
nodes = mesh.getNodes()
nodal_field=np.ones(nodes.shape)*3

# output vector (will be filled by the method "interpolateOnIntegrationPoints")
## to adapt depending on the number of quadrature points (support)
nbr_elem = mesh.getConnectivity(Sup.elemtype).shape[0]
output=np.zeros((nbr_elem,1))

NTF = NodalTensorField("ex_displacement", Sup, nodal_field)

output = aka.ElementTypeMapArrayReal()
output.initialize(mesh, nb_component=1)
NTF.evalOnQuadraturePoints(output)

print(output(aka._segment_2))