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
Sup = Support(elem_filter, fem, spatial_dimension, elem_type)
######################################################################
field_dimension = 1
tol = 10e-8
# Test of GenericOperator :
## comparison between GenericOperator and matmul
## (case of Mass matrix in 1D with 2 elements segment2)

Ngroup = N(Sup,field_dimension)
#GenericOP :
op = GenericOperator("ki", "kj", final = "ij")
using_generic_op = op(Ngroup,Ngroup).evalOnQuadraturePoints()
#@
using_matmul=(transpose(Ngroup)@Ngroup).evalOnQuadraturePoints()

print("Mass matrix using GenericOperator :")
print(using_generic_op)
print("Mass matrix using matmul :")
print(using_matmul)

#Assert that the result is similar
np.testing.assert_allclose(using_generic_op, using_matmul, atol=tol, err_msg="Problem in GenericOperator !")

## comparison between GenericOperator and transpose operation
#GenericOP :
op2 = GenericOperator("ij", final = "ji")
using_generic_op2 = op2(Ngroup).evalOnQuadraturePoints()
#transpose() :
using_transpose = transpose(Ngroup).evalOnQuadraturePoints()

print("Ngroup transpose using GenericOp :")
print(using_generic_op2)
print("Ngroup transpose using transpose :")
print(using_transpose)

#Assert that the result is similar
np.testing.assert_allclose(using_generic_op2, using_transpose, atol=tol, err_msg="Problem in GenericOperator !")