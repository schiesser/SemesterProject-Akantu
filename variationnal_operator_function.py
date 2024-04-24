import akantu as aka
import numpy as np

class Support:
    def __init__(self, elem_filter, fem, spatial_dimension, elem_type, ghost_type):
        self.elem_filter = elem_filter #akaArray
        self.fem = fem
        self.spatial_dimension = spatial_dimension
        self.elem_type = elem_type
        self.ghost_type = ghost_type


class TensorField:
    def __init__(self, name, support):
        self.name = name
        self.support = support

    def evalOnQuadraturePoints(self):
        pass
    
    def getFieldDimension(self):
        pass

    def transpose(self):
        raise NotImplementedError

    def __mul__(self, f):
        return Multiplication(self, f)
    
    def __rmul__(self,f):
        return self.__mul__(f)
    
    def __add__(self, f):
        return Addition(self, f)
    
    def __radd__(self, f):
        return self.__add__(f)
    
    def __sub__(self, f):
        return Substraction(self, f)
    
    def __rsub__(self, f):
        return self.__sub__(f)
    
    def __colon__(self, f):
        return Contraction(self, f)
    
    def __rcolon__(self, f):
        return self.__colon__(f)
 
    
class Operator(TensorField):

    def __init__(self, *args ):

        if len(args) == 2:
            self.first = args[0]
            self.second = args[1]
        elif len(args) == 1:
            self.first = args[0]
            self.second = None
            
        self.support = self.first.support

        self.value_integration_points = None

    def getFieldDimension(self):
        return self.value_integration_points.shape[1] #ok si utilisé après un "evalOnQua..."


class Addition(Operator):

    def __init__(self, f1, f2):

        super().__init__(f1, f2)

        if isinstance(f2, (int, float)):
            self.name = "("+ f1.name + ".ConstantAddition"+")"
        elif isinstance(f2, TensorField):
            self.name = "("+ f1.name + " + " + f2.name + ")"
    
    def evalOnQuadraturePoints(self):

        self.first.evalOnQuadraturePoints()

        if isinstance(self.second, (int, float)):
            self.value_integration_points = self.first.value_integration_points + self.second
        
        elif isinstance(self.second, TensorField):
            self.second.evalOnQuadraturePoints()
            self.value_integration_points = self.first.value_integration_points + self.second.value_integration_points


class Substraction(Operator):

    def __init__(self, f1, f2):

        super().__init__(f1, f2)

        if isinstance(f2, (int, float)):
            self.name = "("+f1.name + ".ConstantSubstraction"+")"
        elif isinstance(f2, TensorField):
            self.name = "("+ f1.name + " - " + f2.name + ")"

    def evalOnQuadraturePoints(self):

        self.first.evalOnQuadraturePoints()

        if isinstance(self.second, (int, float)):
            self.value_integration_points = self.first.value_integration_points - self.second
        
        elif isinstance(self.second, TensorField):
            self.second.evalOnQuadraturePoints()
            self.value_integration_points = self.first.value_integration_points - self.second.value_integration_points


class Multiplication(Operator):

    def __init__(self, f1, f2):

        super().__init__(f1, f2)

        if isinstance(f2, (int, float)):
            self.name = "("+f1.name + ".ConstantMultiplication"+")"
        elif isinstance(f2, TensorField):
            self.name = "("+ f1.name + " * " + f2.name + ")"

    def evalOnQuadraturePoints(self):

        self.first.evalOnQuadraturePoints()

        if isinstance(self.second, (int, float)):
            self.value_integration_points = self.first.value_integration_points * self.second
        
        elif isinstance(self.second, TensorField):
            self.second.evalOnQuadraturePoints()
            self.value_integration_points = self.first.value_integration_points * self.second.value_integration_points


class NodalTensorField(TensorField):
    def __init__(self, name, support, nodal_field):
        super().__init__(name, support)
        self.nodal_field = nodal_field
        self.value_integration_points = None
        
    def getFieldDimension(self):
        return self.nodal_field.shape[1]

    def evalOnQuadraturePoints(self):
        nb_integration_points = self.support.fem.getNbIntegrationPoints(self.support.elem_type)
        mesh = self.support.fem.getMesh()
        nb_element = mesh.getConnectivity(self.support.elem_type).shape[0]

        self.value_integration_points = np.zeros((nb_integration_points*nb_element,self.nodal_field.shape[1])) #dimension : nbr quad point x field dimension

        self.support.fem.interpolateOnIntegrationPoints(
        self.nodal_field, self.value_integration_points, self.value_integration_points.shape[1], self.support.elem_type)
        

class IntegrationPointTensorField(TensorField):
    def evalOnQuadraturePoints(self):
        raise NotImplementedError


class Contraction(Operator):
    #A modifier
    def __init__(self, f1, f2):
        super().__init__(f1, f2)
        #check dimension ?

    def evalOnQuadraturePoints(self):
        raise NotImplementedError


class ShapeField(TensorField):
    def __init__(self, support):
        super().__init__("shape_function", support)
        #attention 1D
        self.conn = support.fem.getMesh().getConnectivities()(support.elem_type)
        self.nb_elem = self.conn.shape[1] # plus tard : nbr quadrature points !
        nb_nodes = self.support.fem.getMesh().getNbNodes() # plus tard encore à mulitpliter par dimension !
        self.value_integration_points = np.zeros((self.nb_elem, nb_nodes))

    def getFieldDimension(self):
        return 3 #nbr de noeud * dimension (selon la sortie voulue !)
    
    def evalOnQuadraturePoints(self):

        shapes_derivatives = self.support.fem.getShapes(self.support.elem_type)

        for i in range(self.nb_elem):
            self.value_integration_points[i,self.conn[i,:]]=shapes_derivatives[i,:]




class GradientOperator(Operator):
    def __init__(self, f1):
        super().__init__(f1) #N

        self.name = "Gradient(" + f1.name + ")"

    def evalOnQuadraturePoints(self):

        if isinstance(self.first, ShapeField):

            self.value_integration_points = self.support.fem.getShapesDerivatives(
            self.support.elem_type)

        '''
        elif isinstance(self.first, ):
            
            raise NotImplementedError'''


class FieldIntegrator:
    @staticmethod
    def integrate(field):
        
        support=field.support
        
        field.evalOnQuadraturePoints()

        field_dim= field.getFieldDimension()
        mesh=support.fem.getMesh()
        nb_element = mesh.getConnectivity(support.elem_type).shape[0]

        nb_integration_points = support.fem.getNbIntegrationPoints(support.elem_type)
        result_integration = np.zeros((nb_element*nb_integration_points, field_dim ))
        
        support.fem.integrate(field.value_integration_points,result_integration,field_dim, support.elem_type)

        integration=np.sum(result_integration,axis=0)
        #integration = result_integration

        return integration
    
