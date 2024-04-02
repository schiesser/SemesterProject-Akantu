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
nodal_field=np.ones((nodes.shape[0],nodes.shape[1]+1))*3
nodal_field[0,0]=2 #première coordonnée : numérotation du noeud; deuxième coordonnée : selon dimensions (ici champ 2D)
nodal_field[2,0]=1

print("nodal_field testé :")
print(nodal_field)
print("avec les connections :")
print(conn)

NTF = NodalTensorField("ex_displacement", Sup, nodal_field)

nb_integration_points = fem.getNbIntegrationPoints(Sup.elem_type)
nb_element = mesh.getConnectivity(Sup.elem_type).shape[0]
value_integration_points=np.zeros((nb_integration_points*nb_element,NTF.getFieldDimension())) #dimension : nbr quad point x field dimension

'''
Si element type map array :
#value_integration_points = aka.ElementTypeMapArrayReal()
#value_integration_points.initialize(mesh, nb_component=1)
'''

NTF.evalOnQuadraturePoints(value_integration_points)

'''
Si element type map array :
value_on_quadpoints=value_on_quadpoints(aka._segment_2)
'''

print("valeurs aux points de quadrature du support")
print(value_integration_points)

# Integrate

## deplacement
integration_depl=FieldIntegrator.integrate(NTF, Sup, mesh)
print("Integration du déplacement: ")
print(integration_depl)

'''
## gradient deplacement
derivative_shapes=Sup.fem.getShapesDerivatives(Sup.elem_type)

extanded_derivative_shapes=np.tile(derivative_shapes,(NTF.getFieldDimension(),1,1))
nodalfieldmod=np.swapaxes(nodal_field[conn],0,-1)
grad_depl=np.sum(np.multiply(nodalfieldmod,extanded_derivative_shapes),axis=-1)
print("gradient associé au champ de deplacement :")
print(grad_depl)
'''
