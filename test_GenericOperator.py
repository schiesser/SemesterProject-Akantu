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
######################################################################
# Début des tests :


#print(Sup.fem.getMesh().getConnectivity(Sup.elem_type))
#print(Sup.fem.getMesh().getNbNodes())

field_dimension = 2
Ngroup = N(Sup,field_dimension)


op = GenericOperator("ik", "jk", final = "ij")
test = op(Ngroup,Ngroup).evalOnQuadraturePoints()

"""
NcN = Contraction(Ngroup, Ngroup)
test = NcN("ijkl, ijml -> km").evalOnQuadraturePoints()
"""

print(test)
