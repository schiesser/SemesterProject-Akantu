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
ghost_type = aka.GhostType(1) #peu importe pour le moment
Sup = Support(elem_filter, fem, spatial_dimension, elemtype, ghost_type)

## Interpolation

# create a field  
nodes = mesh.getNodes()
nodal_field=np.ones(nodes.shape)*3
nodal_field[0,0]=2 #première coordonnée : numérotation du noeud; deuxième coordonnée : selon dimensions (ici 1D)
nodal_field[2,0]=1
print("nodal_field testé :")
print(nodal_field)
print("avec les connections :")
print(conn)

'''
Pour l'interpolation avec la première méthode de "interpolateOnIntegrationPoints", exception resize temporary array
# output vector (will be filled by the method "interpolateOnIntegrationPoints")
## to adapt depending on the number of quadrature points (support)
nbr_elem = mesh.getConnectivity(Sup.elemtype).shape[0]
output=np.zeros((nbr_elem,1)) # avec 1 point de quadrature par élément
'''

#interpolation (avec 2ème méthode de "interpolateOnIntegrationPoints")
NTF = NodalTensorField("ex_displacement", Sup, nodal_field)

output = aka.ElementTypeMapArrayReal()
output.initialize(mesh, nb_component=1)
NTF.evalOnQuadraturePoints(output)

value_on_quadpoints=output(aka._segment_2)
print("valeurs aux point de quadrature du support")
print(value_on_quadpoints)

'''
# Operateur gradient
# bonne taille output selon le support (par son maillage), ici matrice mais à changer ?
nb_element = mesh.getConnectivity(Sup.elemtype).shape[0]
nb_nodes_per_element = mesh.getConnectivity(Sup.elemtype).shape[1] #valable pour segment2
output = np.zeros((nb_element,nb_nodes_per_element))
grad_op=GradientOperator(Sup)
grad_op.evalOnQuadraturePoints(output)
print(output)
'''
# Gradient du déplacement éavlué aux points de quadrature(test fonction integrate)
# (valable car interpolation linéaire)
diff_nodal_value=nodal_field[conn][:,1,:]-nodal_field[conn][:,0,:]
element_length=abs(nodes[conn][:,0,:]-nodes[conn][:,1,:])
gradu=diff_nodal_value/element_length
gradu=gradu.reshape(value_on_quadpoints.shape)
print("gradients du dépl aux points de quadrature :")
print(gradu)


Field_int=FieldIntegrator()
print(Field_int.integrate(NTF, Sup, mesh))
