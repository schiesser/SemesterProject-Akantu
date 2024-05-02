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
    
    def __xor__(self, f):
        return Contraction(self, f)
    
    def __rxor__(self, f):
        return self.__xor__(f)
 
    
class Operator(TensorField):

    def __init__(self, *args ):

        if len(args) == 2:
            self.first = args[0]
            self.second = args[1]
        elif len(args) == 1:
            self.first = args[0]
            self.second = None
            
        self.support = self.first.support

    def getFieldDimension(self):
        return self.value_integration_points.shape[1] #ok si utilisé après un "evalOnQua..."


class Addition(Operator):

    def __init__(self, f1, f2):

        super().__init__(f1, f2)

        if isinstance(f2, (int, float)):
            self.name = "("+ f1.name + ".ConstantAddition"+")"
        elif isinstance(f2, TensorField):
            self.name = "("+ f1.name + " + " + f2.name + ")"
        
        self.value_integration_points = np.zeros(f1.value_integration_points.shape)
    
    def evalOnQuadraturePoints(self):

        self.first.evalOnQuadraturePoints()

        if isinstance(self.second, (int, float)):
            self.value_integration_points = self.first.value_integration_points + self.second
        
        elif isinstance(self.second, TensorField):
            self.second.evalOnQuadraturePoints()
            self.value_integration_points = self.first.value_integration_points + self.second.value_integration_points
        
        return self.value_integration_points


class Substraction(Operator):

    def __init__(self, f1, f2):

        super().__init__(f1, f2)

        if isinstance(f2, (int, float)):
            self.name = "("+f1.name + ".ConstantSubstraction"+")"
        elif isinstance(f2, TensorField):
            self.name = "("+ f1.name + " - " + f2.name + ")"

        self.value_integration_points = np.zeros(f1.value_integration_points.shape)

    def evalOnQuadraturePoints(self):

        self.first.evalOnQuadraturePoints()

        if isinstance(self.second, (int, float)):
            self.value_integration_points = self.first.value_integration_points - self.second
        
        elif isinstance(self.second, TensorField):
            self.second.evalOnQuadraturePoints()
            self.value_integration_points = self.first.value_integration_points - self.second.value_integration_points

        return self.value_integration_points
    

class Multiplication(Operator):

    def __init__(self, f1, f2):

        super().__init__(f1, f2)

        if isinstance(f2, (int, float)):
            self.name = "("+f1.name + ".ConstantMultiplication"+")"
        elif isinstance(f2, TensorField):
            self.name = "("+ f1.name + " * " + f2.name + ")"

        self.value_integration_points = np.zeros(f1.value_integration_points.shape)

    def evalOnQuadraturePoints(self):

        self.first.evalOnQuadraturePoints()

        if isinstance(self.second, (int, float)):
            self.value_integration_points = self.first.value_integration_points * self.second
        
        elif isinstance(self.second, TensorField):
            self.second.evalOnQuadraturePoints()
            self.value_integration_points = self.first.value_integration_points * self.second.value_integration_points
        
        return self.value_integration_points
    

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
        
        return self.value_integration_points
    

class IntegrationPointTensorField(TensorField):
    def evalOnQuadraturePoints(self):
        raise NotImplementedError


class Contraction(Operator):
    #A modifier
    def __init__(self, f1, f2):
        super().__init__(f1, f2)
        
        #check dimension !; provisoirement :
        self.conn = self.support.fem.getMesh().getConnectivities()(self.support.elem_type)
        self.nb_elem = self.conn.shape[0]
        '''
        nb_nodes = self.support.fem.getMesh().getNbNodes() # plus tard encore à mulitpliter par dimension !
        self.value_integration_points = np.zeros((self.nb_elem, nb_nodes, nb_nodes))
        '''

    def evalOnQuadraturePoints(self):
        self.first.evalOnQuadraturePoints()
        self.second.evalOnQuadraturePoints()

        a = self.first.value_integration_points
        b = self.second.value_integration_points

        res_contraction = np.einsum('qi,qj->qij', a, b) # à géneraliser selon la dimension
        self.value_integration_points = res_contraction
        
        nb_nodes = self.support.fem.getMesh().getNbNodes()
        spatial_dimension = self.support.fem.getMesh().getSpatialDimension(self.support.elem_type)
        self.value_integration_points = res_contraction.reshape(self.nb_elem, spatial_dimension*spatial_dimension*nb_nodes*nb_nodes) #encore à modifier selon quadrature point ?

        return self.value_integration_points
    

class ShapeField(TensorField):
    def __init__(self, support):
        super().__init__("shape_function", support)
        self.NbIntegrationPoints=support.fem.getNbIntegrationPoints(support.elem_type)
        self.conn = support.fem.getMesh().getConnectivities()(support.elem_type)
        self.nb_elem = self.conn.shape[0]
        self.nb_nodes_per_elem = self.conn.shape[1]
        self.nb_nodes = self.support.fem.getMesh().getNbNodes()

        self.value_integration_points = np.zeros((self.nb_elem, self.NbIntegrationPoints, support.spatial_dimension, self.nb_nodes_per_elem * support.spatial_dimension))
    
    def getFieldDimension(self):
        return self.value_integration_points.shape[-1]
    
    def evalOnQuadraturePoints(self):

        value_integration_points_without_dim_extension = np.zeros((self.nb_elem, self.NbIntegrationPoints,1, self.nb_nodes_per_elem))
        shapes = self.support.fem.getShapes(self.support.elem_type)
        shapes = shapes.reshape((value_integration_points_without_dim_extension.shape))
        
        for i in range(self.support.spatial_dimension):
            self.value_integration_points[:,:,i::self.support.spatial_dimension,i::self.support.spatial_dimension]=shapes

        temp = shapes.reshape((self.nb_elem,self.NbIntegrationPoints*self.nb_nodes_per_elem))
        
        return temp
    

class GradientOperator(Operator):
    def __init__(self, f1):
        super().__init__(f1)
        self.name = "Gradient(" + f1.name + ")"

        if isinstance(self.first, ShapeField):
            self.conn = self.support.fem.getMesh().getConnectivities()(self.support.elem_type)
            self.nb_elem = self.conn.shape[0]
            self.nb_nodes = self.support.fem.getMesh().getNbNodes()
            self.NbIntegrationPoints=self.support.fem.getNbIntegrationPoints(self.support.elem_type)
            self.nb_nodes_per_elem = self.conn.shape[1]

            self.value_integration_points = np.zeros((self.nb_elem, self.NbIntegrationPoints, self.support.spatial_dimension, self.nb_nodes_per_elem * self.support.spatial_dimension))

    def getFieldDimension(self):
        test = self.nb_nodes_per_elem * self.support.spatial_dimension * self.support.spatial_dimension
        return test
    
    def evalOnQuadraturePoints(self):

        if isinstance(self.first, ShapeField):
            
            value_integration_points_without_dim_extension = np.zeros((self.nb_elem, self.NbIntegrationPoints,1, self.nb_nodes_per_elem* self.support.spatial_dimension))
            shapes = self.support.fem.getShapesDerivatives(self.support.elem_type)
            print("shapes shape :")
            print(shapes.shape)
            shapes = shapes.reshape((value_integration_points_without_dim_extension.shape))
            temp = shapes.reshape((self.nb_elem,self.NbIntegrationPoints*self.nb_nodes_per_elem* self.support.spatial_dimension))
            
        return temp


class FieldIntegrator:
    @staticmethod
    def integrate(field):
        
        support=field.support
        
        value_integration_points=field.evalOnQuadraturePoints()

        field_dim= field.getFieldDimension()
        mesh=support.fem.getMesh()
        nb_element = mesh.getConnectivity(support.elem_type).shape[0]

        nb_integration_points = support.fem.getNbIntegrationPoints(support.elem_type)
        result_integration = np.zeros((nb_element*nb_integration_points, field_dim ))
        
        support.fem.integrate(value_integration_points,result_integration,field_dim, support.elem_type)
        integration=np.sum(result_integration,axis=0)

        return integration
    
class FieldIntegrator2:
    @staticmethod
    def integrate(field):
        
        support=field.support
        
        value_integration_points=field.evalOnQuadraturePoints()

        field_dim= field.getFieldDimension()
        mesh=support.fem.getMesh()
        nb_element = mesh.getConnectivity(support.elem_type).shape[0]

        result_integration = np.zeros((nb_element, field.nb_nodes_per_elem))
        
        support.fem.integrate(value_integration_points,result_integration,field_dim, support.elem_type)
        integration = result_integration

        return integration
    
    def assembly(field, result_integration):

        if isinstance(field, ShapeField):
                conn = field.conn
                nb_elem = field.nb_elem
                nb_nodes = np.max(conn)+1 #+1 car noeud 0
                assembled_result = np.zeros((nb_nodes))
                for i in range(nb_elem):
                    assembled_result[conn[i,:]]+=result_integration[i,:]

        elif isinstance(field, GradientOperator):
                conn = field.conn
                nb_elem = field.nb_elem
                nb_nodes = np.max(conn)+1 #+1 car noeud 0
                assembled_result = np.zeros((nb_nodes))
                for i in range(nb_elem):
                    assembled_result[conn[i,:]]+=result_integration[i,:]

        return assembled_result
    
    