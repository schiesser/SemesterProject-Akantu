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
elemtype = aka._segment_2
ghost_type = aka.GhostType(1) #peu importe pour le moment
Sup = Support(elem_filter, fem, spatial_dimension, elemtype, ghost_type)

# Interpolation

## create a field  
nodes = mesh.getNodes()


### Cas d'un champ scalaire :
nodal_field=np.ones(nodes.shape)*3
nodal_field[0,0]=2 #première coordonnée : numérotation du noeud; deuxième coordonnée : selon dimensions (ici 1D)
nodal_field[2,0]=1

'''
### Cas d'un champ vectoriel :
nodal_field=np.ones((nodes.shape[0],nodes.shape[1]+1))*3
nodal_field[0,0]=2 #première coordonnée : numérotation du noeud; deuxième coordonnée : selon dimensions (ici champ 2D)
nodal_field[2,0]=1
'''

print("nodal_field testé :")
print(nodal_field)
print("avec les connections :")
print(conn)


# Interpolation (avec 2ème méthode de "interpolateOnIntegrationPoints")
NTF = NodalTensorField("ex_displacement", Sup, nodal_field)

numberIntegrationPoints = fem.getNbIntegrationPoints(Sup.elemtype)
nb_element = mesh.getConnectivity(Sup.elemtype).shape[0]
value_on_quadpoints=np.zeros((numberIntegrationPoints*nb_element,NTF.getFieldDimension())) #dimension : nbr quad point x field dimension

'''
Si element type map array :
#output = aka.ElementTypeMapArrayReal()
#output.initialize(mesh, nb_component=1)
'''

NTF.evalOnQuadraturePoints(value_on_quadpoints)

'''
Si element type map array :
value_on_quadpoints=value_on_quadpoints(aka._segment_2)
'''

print("valeurs aux points de quadrature du support")
print(value_on_quadpoints)
print(value_on_quadpoints.shape)


# Integrate

## deplacement
integrationDepl=FieldIntegrator.integrate(NTF, Sup, mesh)
print("Integration : ")
print(integrationDepl)

## gradient deplacement
derivativeShapes=Sup.fem.getShapesDerivatives(Sup.elemtype)

extandedDerivativeShapes=np.tile(derivativeShapes,(NTF.getFieldDimension(),1,1))
nodalfieldmod=np.swapaxes(nodal_field[conn],0,-1)
graddispl=np.sum(np.multiply(nodalfieldmod,extandedDerivativeShapes),axis=-1)
print("gradient associé au champ de deplacement :")
print(graddispl)


field_dim= NTF.getFieldDimension()
res = np.zeros((nb_element, field_dim ))
Sup.fem.integrate(graddispl,res,field_dim, Sup.elemtype)
print("Résultat de l'intégration")
print(res)