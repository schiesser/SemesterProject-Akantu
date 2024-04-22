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

'''
### Cas d'un champ scalaire :
nodal_field=np.ones(nodes.shape)*3
nodal_field[0,0]=2 #première coordonnée : numérotation du noeud; deuxième coordonnée : selon dimensions (ici 1D)
nodal_field[2,0]=1
'''
### Cas d'un champ vectoriel :
nodal_vector=np.ones((nodes.shape[0],nodes.shape[1]))*3
#nodal_vector[0,0]=2 #première coordonnée : numérotation du noeud; deuxième coordonnée : selon dimensions (ici champ 2D)
#nodal_vector[2,0]=1

'''
print("nodal_field testé :")
print(nodal_vector)
print("avec les connections :")
print(conn)
'''

NTF2 = NodalTensorField("displacement2", Sup, nodal_vector)


## gradient deplacement


field_dim = nodes.shape[1]
nb_element = mesh.getConnectivity(Sup.elem_type).shape[0]
nb_integration_points = Sup.fem.getNbIntegrationPoints(Sup.elem_type)
result_integration = np.zeros((nb_element*nb_integration_points, field_dim ))
nodes_per_elem=2

n = nodal_vector[conn].reshape((nodes_per_elem,nb_element))
der = Sup.fem.getShapesDerivatives(Sup.elem_type)

gradquapoint = np.sum(n*der,axis=1)


gradquapoint =gradquapoint.reshape((nb_integration_points*nb_element,field_dim))
Sup.fem.integrate(gradquapoint,result_integration,field_dim, Sup.elem_type)

print(np.sum(result_integration,axis=0))
'''
print(dir(Sup.fem.getShapesDerivatives(Sup.elem_type)))
print(Sup.fem.getShapesDerivatives(Sup.elem_type))
'''