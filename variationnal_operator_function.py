
class Support:
    def __init__(self, elem_filter, fem, spatial_dimension, elemtype, ghost_type):
        self.elem_filter = elem_filter
        self.fem = fem
        self.spatial_dimension = spatial_dimension
        self.elemtype = elemtype
        self.ghost_type = ghost_type

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

    def evalOnQuadraturePoints(self, output):
        self.support.fem.interpolateOnIntegrationPoints(
            self.nodal_field, output, self.nodal_field.shape[0]*self.nodal_field.shape[1], self.support.elemtype)

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
    def evalOnQuadraturePoints(self, output):
        shapes_derivatives = self.support.fem.getShapesDerivatives(
            self.support.elemtype, self.support.ghost_type
        )
        output[:] = shapes_derivatives

class FieldIntegrator:
    @staticmethod
    def integrate(field, support):
        nb_element = len(support.elem_filter)
        nb_nodes_per_element = Mesh.getNbNodesPerElement(support.elemtype)
        res = np.zeros((nb_element, nb_nodes_per_element * support.spatial_dimension))

        nb_component = field.getNbComponent()

        field_eval = np.array([])
        field.evalOnQuadraturePoints(field_eval)

        support.fem.integrate(field_eval, res, nb_component, support.elemtype,
                              support.ghost_type, support.elem_filter)
        return res