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
        self.value_integration_points = None

    def evalOnQuadraturePoints(self):
        pass
    
    def getFieldDimension(self):
        pass

    def getDimensionForIntegration(self):
        return np.prod(self.value_integration_points.shape[1:])

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


class Addition(Operator):

    def __init__(self, f1, f2):

        super().__init__(f1, f2)

        if isinstance(f2, (int, float)):
            self.name = "("+ f1.name + ".ConstantAddition"+")"
        elif isinstance(f2, TensorField):
            self.name = "("+ f1.name + " + " + f2.name + ")"
        
        self.value_integration_points = np.zeros(f1.value_integration_points.shape)
    
    def evalOnQuadraturePoints(self):

        firstevaluated = self.first.evalOnQuadraturePoints()

        if isinstance(self.second, (int, float)):
            self.value_integration_points = firstevaluated + self.second
        
        elif isinstance(self.second, TensorField):
            secondevaluated = self.second.evalOnQuadraturePoints()
            self.value_integration_points = firstevaluated + secondevaluated
        
        return self.value_integration_points

class transpose(Operator):
    # valide pour des array d'ordre 4 (nb_elem, nb_points_integration, dim1, dim2)
    # transpose dim1 et dim2
    # à généraliser par la suite selon besoins
    def __init__(self,f):

        super().__init__(f)
        self.name = "transpose"+"("+self.first.name+")"
        self.value_integration_points = np.zeros((f.value_integration_points.shape[0],f.value_integration_points.shape[1],f.value_integration_points.shape[3],f.value_integration_points.shape[2]))
    
    def evalOnQuadraturePoints(self):

        firstevaluated = self.first.evalOnQuadraturePoints()

        self.value_integration_points = np.transpose(firstevaluated, axes=(0,1,3,2))
        
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

        firstevaluated = self.first.evalOnQuadraturePoints()

        if isinstance(self.second, (int, float)):
            self.value_integration_points = firstevaluated - self.second
        
        elif isinstance(self.second, TensorField):
            secondevaluated = self.second.evalOnQuadraturePoints()
            self.value_integration_points = firstevaluated - secondevaluated

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

        firstevaluated = self.first.evalOnQuadraturePoints()

        if isinstance(self.second, (int, float)):
            self.value_integration_points = firstevaluated * self.second
        
        elif isinstance(self.second, TensorField):
            secondevaluated = self.second.evalOnQuadraturePoints()
            self.value_integration_points = firstevaluated * secondevaluated
        
        return self.value_integration_points
    

class NodalTensorField(TensorField):
    def __init__(self, name, support, nodal_field):
        super().__init__(name, support)
        self.nodal_field = nodal_field
        
        nb_integration_points = self.support.fem.getNbIntegrationPoints(self.support.elem_type)
        mesh = self.support.fem.getMesh()
        nb_element = mesh.getConnectivity(self.support.elem_type).shape[0]
        self.value_integration_points = np.zeros((nb_integration_points*nb_element,self.nodal_field.shape[1])) #dimension : nbr quad point x field dimension

    def getFieldDimension(self):
        return self.nodal_field.shape[1]

    def getDimensionForIntegration(self):
        return self.nodal_field.shape[1]
    
    def evalOnQuadraturePoints(self):
        self.support.fem.interpolateOnIntegrationPoints(self.nodal_field, self.value_integration_points, self.value_integration_points.shape[1], self.support.elem_type)
        
        return self.value_integration_points
    

class Contraction(Operator):
    #A modifier
    def __init__(self, f1, f2):
        super().__init__(f1, f2)
        
        #Ajouter exceptions de contrôle de dimensions !
        self.value_integration_points = np.zeros(f1.value_integration_points.shape)
    
    def evalOnQuadraturePoints(self):

        firstevaluated = self.first.evalOnQuadraturePoints()
        secondevaluated = self.second.evalOnQuadraturePoints()
        
        self.value_integration_points = np.matmul(firstevaluated,secondevaluated)

        """tests avec utilisation einsum :
        print("shape de a :")
        print(a.shape)
        print("shape de b :")
        print(b.shape)
        res_contraction = np.einsum('ijkl,jlmn->ikmn', a, b)
        res_contraction = np.einsum('qi,qj->qij', a, b) # à géneraliser selon la dimension
        self.value_integration_points = res_contraction
        """

        return self.value_integration_points
    

class ShapeField(TensorField):
    def __init__(self, support):
        super().__init__("shape_function", support)
        self.NbIntegrationPoints=support.fem.getNbIntegrationPoints(support.elem_type)
        self.conn = support.fem.getMesh().getConnectivities()(support.elem_type)
        self.nb_elem = self.conn.shape[0]
        self.nb_nodes_per_elem = self.conn.shape[1]

        self.value_integration_points = np.zeros((self.nb_elem * self.NbIntegrationPoints, self.nb_nodes_per_elem))
    
    def evalOnQuadraturePoints(self):

        self.value_integration_points = self.support.fem.getShapes(self.support.elem_type)
        
        return self.value_integration_points

class N(ShapeField):
    def __init__(self, support, dim_field):
        super().__init__(support)
        self.dim_field = dim_field
        # array qui contient tous les N est de dimension :
        # (nombre éléments, nombre points Gauss, dimension du champ, noeuds par élément x dimension du champ)
        self.value_integration_points = np.zeros((self.nb_elem, self.NbIntegrationPoints, self.dim_field, self.nb_nodes_per_elem * self.dim_field))
    
    def evalOnQuadraturePoints(self):

        N_without_dim_extension = np.zeros((self.nb_elem, self.NbIntegrationPoints,1, self.nb_nodes_per_elem))
        shapes = self.support.fem.getShapes(self.support.elem_type)
        shapes = shapes.reshape((N_without_dim_extension.shape))
        
        for i in range(self.dim_field):
            self.value_integration_points[:,:,i::self.dim_field,i::self.dim_field]=shapes
        
        return self.value_integration_points

class GradientOperator(Operator):
    def __init__(self, f1):
        super().__init__(f1)
        self.name = "Gradient(" + f1.name + ")"

        if isinstance(self.first, ShapeField):
            self.conn = self.first.conn
            self.nb_elem = self.conn.shape[0]
            self.NbIntegrationPoints=self.first.NbIntegrationPoints
            self.nb_nodes_per_elem = self.conn.shape[1]
            self.dim_field = self.first.dim_field

            if self.support.spatial_dimension == 1 :
                self.nb_line = 1

            elif self.support.spatial_dimension == 2 :
                self.nb_line = 3

            self.value_integration_points = np.zeros((self.nb_elem, self.NbIntegrationPoints, self.nb_line, self.nb_nodes_per_elem * self.dim_field))

    def evalOnQuadraturePoints(self):

        if isinstance(self.first, ShapeField):

            B_without_dim_extension = np.zeros((self.nb_elem, self.NbIntegrationPoints,1, self.nb_nodes_per_elem*self.support.spatial_dimension))
            derivatives_shapes = self.support.fem.getShapesDerivatives(self.support.elem_type)
            derivatives_shapes = derivatives_shapes.reshape((B_without_dim_extension.shape))

            if self.support.spatial_dimension == 1:
                for i in range(self.dim_field):
                    self.value_integration_points[:,:,:,i::self.dim_field]=derivatives_shapes

            elif self.support.spatial_dimension == 2:
                for i in range(self.nb_elem):
                    for j in range(self.dim_field):
                        if self.dim_field == 1:
                            self.value_integration_points[i,:,0,0::self.dim_field]=derivatives_shapes[i,:,0,:self.nb_nodes_per_elem]
                            self.value_integration_points[i,:,1,0::self.dim_field]=derivatives_shapes[i,:,0,self.nb_nodes_per_elem:]
                            self.value_integration_points[i,:,2,0::self.dim_field]+=derivatives_shapes[i,:,0,:self.nb_nodes_per_elem]
                            self.value_integration_points[i,:,2,0::self.dim_field]+=derivatives_shapes[i,:,0,self.nb_nodes_per_elem:]
                        
                        if self.dim_field == 2:
                            self.value_integration_points[i,:,0,0::self.dim_field]=derivatives_shapes[i,:,0,:self.nb_nodes_per_elem]
                            self.value_integration_points[i,:,1,1::self.dim_field]=derivatives_shapes[i,:,0,self.nb_nodes_per_elem:]
                            self.value_integration_points[i,:,2,1::self.dim_field]=derivatives_shapes[i,:,0,:self.nb_nodes_per_elem]
                            self.value_integration_points[i,:,2,0::self.dim_field]=derivatives_shapes[i,:,0,self.nb_nodes_per_elem:]
        
        return self.value_integration_points


class FieldIntegrator:
    @staticmethod
    # Valide pour des cas où on veut intégrer un array de dimension 2 (par exemple les NodalTensorField)
    def integrate(field):
        
        support = field.support
        
        value_integration_points=field.evalOnQuadraturePoints()

        int_dim = field.getDimensionForIntegration()
        mesh=support.fem.getMesh()
        nb_element = mesh.getConnectivity(support.elem_type).shape[0]
        nb_integration_points = support.fem.getNbIntegrationPoints(support.elem_type)

        result_integration = np.zeros((nb_element*nb_integration_points, int_dim ))
        
        support.fem.integrate(value_integration_points,result_integration,int_dim, support.elem_type)
        integration=np.sum(result_integration,axis=0)

        return integration
    
class FieldIntegrator2:
    #provisoirement un deuxième classe d'intégration
    # une fois que les tests sont passés, je regrouperai
    @staticmethod
    def integrate(field):
        
        support=field.support
        
        value_integration_points=field.evalOnQuadraturePoints()
        shape_output = value_integration_points.shape


        mesh=support.fem.getMesh()
        nb_element = mesh.getConnectivity(support.elem_type).shape[0]
        
        value_integration_points = value_integration_points.reshape((nb_element,-1))
        int_dim = field.getDimensionForIntegration()
        
        result_integration = np.zeros((nb_element, int_dim))

        support.fem.integrate(value_integration_points,result_integration,int_dim, support.elem_type)
        integration = result_integration.reshape(shape_output)

        return integration
    
class Assembly:
    @staticmethod
    def assemblyK(conn, groupedKlocal, dim1, dim2, field_dim):
        # pour matrice de rigidité globale
        n_elem  = conn.shape[0]
        n_nodes_per_elem = conn.shape[1]
        numEq = np.zeros((n_elem, field_dim*n_nodes_per_elem), dtype=int)
        
        for e in range(n_elem):
            for i in range(n_nodes_per_elem):
                    for j in range(field_dim):
                        numEq[e, field_dim*i+j] = field_dim*conn[e, i]+j

        K = np.zeros((dim1, dim2))

        for e in range(n_elem):

            ddl = numEq[e, :]

            K_locale = groupedKlocal

            for i, gi in enumerate(ddl):
                for j, gj in enumerate(ddl):
                    K[gi, gj] += K_locale[e,0,i, j]
        return K

    def assemblyV(conn, groupedV, dim2, field_dim):
        #pour integration de N ou B
        n_elem  = conn.shape[0]
        n_nodes_per_elem = conn.shape[1]
        numEq = np.zeros((n_elem, field_dim*n_nodes_per_elem), dtype=int)
        
        for e in range(n_elem):
            for i in range(n_nodes_per_elem):
                    for j in range(field_dim):
                        numEq[e, field_dim*i+j] = field_dim*conn[e, i]+j

        V = np.zeros((field_dim, dim2))

        for e in range(n_elem):

            ddl = numEq[e, :]

            V_locale = groupedV

            for i, gi in enumerate(ddl):
                    V[:, gi] += V_locale[e,0,:,i]

        return V
