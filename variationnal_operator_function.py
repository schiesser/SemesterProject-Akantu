import akantu as aka
import numpy as np

class Support:
    def __init__(self, elem_filter, fem, spatial_dimension, elem_type, ghost_type):
        self.elem_filter = elem_filter #ici : akaArray
        self.fem = fem
        self.spatial_dimension = spatial_dimension
        self.elem_type = elem_type
        self.ghost_type = ghost_type

    '''
    def __init__(self, elem_filter, fem, spatial_dimension, ghost_type):
        self.elem_filter = elem_filter #ici : ElementTypeMapArrayInt
        self.fem = fem
        self.spatial_dimension = spatial_dimension
    '''


class TensorField:
    def __init__(self, name, support):
        self.name = name
        self.support = support

    def evalOnQuadraturePoints(self, output):
        pass

    def getNbComponent(self):
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError

    def __mul__(self, f):
        return DotField(self, f)


class NodalTensorField(TensorField):
    def __init__(self, name, support, nodal_field):
        super().__init__(name, support)
        self.nodal_field = nodal_field

    def getFieldDimension(self):
        return self.nodal_field.shape[1]

    def evalOnQuadraturePoints(self, output):
        #help(self.support.fem.interpolateOnIntegrationPoints)
        self.support.fem.interpolateOnIntegrationPoints(
        self.nodal_field, output, output.shape[1], self.support.elem_type)
        #self.support.fem.interpolateOnIntegrationPoints(self.nodal_field, output, self.support.elem_filter)


class IntegrationPointTensorField(TensorField):
    def evalOnQuadraturePoints(self, output):
        raise NotImplementedError


class DotField(TensorField):
    def __init__(self, f1, f2):
        super().__init__(f1.name + "." + f2.name, f1.support)
        self.field1 = f1
        self.field2 = f2

    def evalOnQuadraturePoints(self, output):
        o1 = np.array([])
        o2 = np.array([])
        self.field1.evalOnQuadratureNode(o1)
        self.field2.evalOnQuadratureNode(o2)
        for i in range(len(output)):
            output[i] = o1[i] * o2[i]


class GradientOperator(TensorField):
    def __init__(self, support):
        super().__init__("gradient", support)

    def evalOnQuadraturePoints(self, output):
        shapes_derivatives = self.support.fem.getShapesDerivatives(
            self.support.elemtype)
        
        output[:,:] = shapes_derivatives


class FieldIntegrator:
    @staticmethod
    def integrate(field, support, mesh):
        
        number_integration_points = support.fem.getNbIntegrationPoints(support.elem_type)
        field_dim= field.getFieldDimension()
        nb_element = mesh.getConnectivity(support.elem_type).shape[0]
        field_eval=np.zeros((nb_element*number_integration_points,field_dim))

        #field_eval = aka.ElementTypeMapArrayReal()
        #field_eval.initialize(mesh, nb_component=self.support.spatial_dimension)
        field.evalOnQuadraturePoints(field_eval)

        result_integration = np.zeros((nb_element, field_dim )) #for one quadrature point per elem
        
        #help(support.fem.integrate)
        #abc=support.fem.integrate(field_eval(support."elem_type"),res,field_dim, elemtype,support.elem_type,filter_elements=support.elem_filter)#passer elemfilter en akaArray
        support.fem.integrate(field_eval,result_integration,field_dim, support.elem_type)

        integration=np.sum(result_integration,axis=0)

        return integration
