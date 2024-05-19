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
    
    def __matmul__(self, f):
        return Contraction(self, f)
 
    
class Operator(TensorField):
 
    def __init__(self, *args):
        self.args = args
        self.support = args[0].support
    
    def getFieldDimension(self):
        # Ok pour toutes les opération sur NodalTensorField, 
        # autres cas : override dans classes dérivées
        return self.args[0].getFieldDimension()


class Addition(Operator):

    def __init__(self, f1, f2):
        
        super().__init__(f1, f2)

        if isinstance(f2, (int, float)):
            self.name = "("+ f1.name + ".ConstantAddition"+")"

            if not "NodalTensorField" in f1.name:
                raise TypeError("this operation only works for NodalTensorField")
            
        elif isinstance(f2, TensorField):
            self.name = "("+ f1.name + " + " + f2.name + ")"

            if not "NodalTensorField" in (f1.name and f2.name):
                raise TypeError("this operation only works for NodalTensorField")
        
        self.value_integration_points = np.zeros(f1.value_integration_points.shape)
    
    def evalOnQuadraturePoints(self):

        firstevaluated = self.args[0].evalOnQuadraturePoints()

        if isinstance(self.args[1], (int, float)):
            self.value_integration_points = firstevaluated + self.args[1]
        
        elif isinstance(self.args[1], TensorField):
            secondevaluated = self.args[1].evalOnQuadraturePoints()

            if np.all(firstevaluated.shape != secondevaluated):
                raise TypeError("the 2 evaluated 'value_integration_point' don't have the same shape : can't do this operation")

            self.value_integration_points = firstevaluated + secondevaluated
        
        return self.value_integration_points
    
    
class transpose(Operator):
    # transpose les 2 dernières dimensions de l'array : "value_integration_point"; pensé pour les type GradientOperator ou N. 
    def __init__(self,f):

        if not isinstance(f, (GradientOperator, N)):
            raise TypeError("Be careful if you want to transpose an object different from grad(N) or N. It transposes the last 2 dimensions of an array. Other possibility : use Contraction class with particular subscripts: it uses einsum form numpy.")
        
        super().__init__(f)
        self.name = "transpose"+"("+f.name+")"
        self.value_integration_points = np.transpose(f.value_integration_points, axes=list(range(len(f.value_integration_points.shape) - 2)) + [-1, -2])    
    
    def evalOnQuadraturePoints(self):

        firstevaluated = self.args[0].evalOnQuadraturePoints()

        self.value_integration_points = np.transpose(firstevaluated, axes=list(range(len(firstevaluated.shape) - 2)) + [-1, -2])        
        return self.value_integration_points

    def getFieldDimension(self):
        #pas fait pour être intégré "brut" :
        #goal : intégrer un objet retourné par une opération entre plusieurs objet dont un est transposé
        #utilise donc le getFieldDim de la contraction par exemple, mais jamais celui-ci directement.
        raise NotImplementedError("Not possible to directly integrate a transpose.")

class Substraction(Operator):

    def __init__(self, f1, f2):
        
        super().__init__(f1, f2)

        if isinstance(f2, (int, float)):
            self.name = "("+f1.name + ".ConstantSubstraction"+")"

            if not "NodalTensorField" in f1.name:
                raise TypeError("this operation only works for NodalTensorField")
            
        elif isinstance(f2, TensorField):
            self.name = "("+ f1.name + " - " + f2.name + ")"

            if not "NodalTensorField" in (f1.name and f2.name):
                raise TypeError("this operation only works for NodalTensorField")

        self.value_integration_points = np.zeros(f1.value_integration_points.shape)

    def evalOnQuadraturePoints(self):

        firstevaluated = self.args[0].evalOnQuadraturePoints()

        if isinstance(self.args[1], (int, float)):
            self.value_integration_points = firstevaluated - self.args[1]
        
        elif isinstance(self.args[1], TensorField):           
            secondevaluated = self.args[1].evalOnQuadraturePoints()

            if np.all(firstevaluated.shape != secondevaluated):
                raise TypeError("the 2 evaluated 'value_integration_point' don't have the same shape : can't do this operation")
            
            self.value_integration_points = firstevaluated - secondevaluated

        return self.value_integration_points
    

class Multiplication(Operator):

    def __init__(self, f1, f2):
        
        super().__init__(f1, f2)

        if isinstance(f2, (int, float)):
            self.name = "("+f1.name + ".ConstantMultiplication"+")"

            if not "NodalTensorField" in f1.name:
                raise TypeError("this operation only works for NodalTensorField")
            
        elif isinstance(f2, TensorField):
            self.name = "("+ f1.name + " * " + f2.name + ")"

            if not "NodalTensorField" in (f1.name and f2.name):
                raise TypeError("this operation only works for NodalTensorField")

        self.value_integration_points = np.zeros(f1.value_integration_points.shape)

    def evalOnQuadraturePoints(self):

        firstevaluated = self.args[0].evalOnQuadraturePoints()

        if isinstance(self.args[1], (int, float)):
            self.value_integration_points = firstevaluated * self.args[1]
        
        elif isinstance(self.args[1], TensorField):
            secondevaluated = self.args[1].evalOnQuadraturePoints()

            if np.all(firstevaluated.shape != secondevaluated):
                raise TypeError("the 2 evaluated 'value_integration_point' don't have the same shape : can't do this operation")
            
            self.value_integration_points = firstevaluated * secondevaluated
        
        return self.value_integration_points
    

class NodalTensorField(TensorField):
    def __init__(self, name, support, nodal_field):
        super().__init__("NodalTensorField("+name+")", support)
        self.nodal_field = nodal_field
        
        nb_integration_points = self.support.fem.getNbIntegrationPoints(self.support.elem_type)
        mesh = self.support.fem.getMesh()
        nb_element = mesh.getConnectivity(self.support.elem_type).shape[0]
        self.value_integration_points = np.zeros((nb_integration_points*nb_element,self.nodal_field.shape[1])) #dimension : nbr quad point x field dimension

    def getFieldDimension(self):
        return self.nodal_field.shape[1]
    
    def evalOnQuadraturePoints(self):
        self.support.fem.interpolateOnIntegrationPoints(self.nodal_field, self.value_integration_points, self.value_integration_points.shape[1], self.support.elem_type)
        
        return self.value_integration_points
   
class GenericOperator:

    def __init__(self, *args, final=None):
        if final==None:
            raise ValueError("please give the indices of result array")
        self.subscripts_for_summation = ','.join(["xy"+i for i in args])
        self.subscripts_for_summation+= "->" + "xy" + final

    def __call__(self,*args):

        return Contraction(*((self.subscripts_for_summation,) + args))


class Contraction(Operator):
    def __init__(self, *args):

        if isinstance(args[0], str):
            self.subscripts_for_summation = args[0]
            super().__init__(*args[1:])
        else :
            self.subscripts_for_summation = None
            super().__init__(*args)

        self.value_integration_points = None #donner bonne dimension !

    def evalOnQuadraturePoints(self):
        fieldevaluated = [tensor_field.evalOnQuadraturePoints() for tensor_field in self.args]

        if self.subscripts_for_summation is None:

            self.value_integration_points = np.matmul(*fieldevaluated)
        else :
        
            self.value_integration_points = np.einsum(self.subscripts_for_summation, *fieldevaluated)
        
        return self.value_integration_points
    
    def getFieldDimension(self):
        return np.prod(self.value_integration_points.shape[-2:])


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
    
    def getFieldDimension(self):
        # Pas fait pour être intégrer directement
        raise NotImplementedError
    
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

    def getFieldDimension(self):
        return np.prod(self.value_integration_points.shape[-2:])

class ConstitutiveLaw(N):
    def __init__(self,nb_element, D):
        self.D = D
        self.nb_elem = nb_element
        self.value_integration_points = np.zeros((nb_element,1,3,3))#2D

    def evalOnQuadraturePoints(self):
        D = self.D.reshape((1,1,3,3))
        self.value_integration_points = np.tile(D, (self.nb_elem, 1, 1, 1))
        return self.value_integration_points
    
class GradientOperator(Operator):
    def __init__(self, f1):
        super().__init__(f1)
        self.name = "Gradient(" + f1.name + ")"

        self.conn = f1.conn
        self.nb_elem = self.conn.shape[0]
        self.NbIntegrationPoints=f1.NbIntegrationPoints
        self.nb_nodes_per_elem = self.conn.shape[1]
        self.dim_field = 1

        if isinstance(f1, N):
            
            self.dim_field = f1.dim_field

            if self.support.spatial_dimension == 1 :
                self.nb_line = 1 *self.dim_field

            elif self.support.spatial_dimension == 2 :
                self.nb_line = 3

            self.value_integration_points = np.zeros((self.nb_elem, self.NbIntegrationPoints, self.nb_line, self.nb_nodes_per_elem * self.dim_field))
        
        elif isinstance(f1, ShapeField):
            
            self.value_integration_points = np.zeros((self.nb_elem, self.NbIntegrationPoints, self.support.spatial_dimension, self.nb_nodes_per_elem * self.dim_field))
            

        else : 
            raise NotImplementedError("gradient is implemented only for a shapefield")
          
    
    def evalOnQuadraturePoints(self):

        if isinstance(self.args[0], N):

            B_without_dim_extension = np.zeros((self.nb_elem, self.NbIntegrationPoints,1, self.nb_nodes_per_elem*self.support.spatial_dimension))
            derivatives_shapes = self.support.fem.getShapesDerivatives(self.support.elem_type)
            derivatives_shapes = derivatives_shapes.reshape((B_without_dim_extension.shape))

            if self.support.spatial_dimension == 1:
                for i in range(self.dim_field):
                    self.value_integration_points[:,:,i::self.dim_field,i::self.dim_field]=derivatives_shapes

            elif self.support.spatial_dimension == 2:
                for i in range(self.nb_elem):
                    for j in range(self.dim_field):
                        if self.dim_field == 1: # ce cas est problématique lors de la contraction !
                            self.value_integration_points[i,:,0,0::self.dim_field]=derivatives_shapes[i,:,0,:self.nb_nodes_per_elem]
                            self.value_integration_points[i,:,1,0::self.dim_field]=derivatives_shapes[i,:,0,self.nb_nodes_per_elem:]
                            self.value_integration_points[i,:,2,0::self.dim_field]+=derivatives_shapes[i,:,0,:self.nb_nodes_per_elem]
                            self.value_integration_points[i,:,2,0::self.dim_field]+=derivatives_shapes[i,:,0,self.nb_nodes_per_elem:]
                        
                        if self.dim_field == 2:
                            self.value_integration_points[i,:,0,0::self.dim_field]=derivatives_shapes[i,:,0,:self.nb_nodes_per_elem]
                            self.value_integration_points[i,:,1,1::self.dim_field]=derivatives_shapes[i,:,0,self.nb_nodes_per_elem:]
                            self.value_integration_points[i,:,2,1::self.dim_field]=derivatives_shapes[i,:,0,:self.nb_nodes_per_elem]
                            self.value_integration_points[i,:,2,0::self.dim_field]=derivatives_shapes[i,:,0,self.nb_nodes_per_elem:]
        
        elif isinstance(self.args[0], ShapeField):

            B_without_dim_extension = np.zeros((self.nb_elem, self.NbIntegrationPoints,1,self.support.spatial_dimension * self.nb_nodes_per_elem))
            derivatives_shapes = self.support.fem.getShapesDerivatives(self.support.elem_type)
            derivatives_shapes = derivatives_shapes.reshape((B_without_dim_extension.shape))
            
            for i in range(self.support.spatial_dimension):
                for j in range(self.dim_field):
                    self.value_integration_points[:,:,i,j::self.dim_field]=derivatives_shapes[:,:,0,i*self.nb_nodes_per_elem:self.nb_nodes_per_elem*(i+1)]
            
        return self.value_integration_points
    
    def getFieldDimension(self):

        return np.prod(self.value_integration_points.shape[-2:])
    

class FieldIntegrator:
    @staticmethod
    def integrate(field):
        
        support = field.support
        
        value_integration_points=field.evalOnQuadraturePoints()
        int_dim = field.getFieldDimension()

        value_integration_points = value_integration_points.reshape((-1,int_dim))

        mesh=support.fem.getMesh()

        nb_element = mesh.getConnectivity(support.elem_type).shape[0]

        result_integration = np.zeros((nb_element, int_dim ))
        
        NbIntegrationPoints=field.support.fem.getNbIntegrationPoints(support.elem_type)

        if value_integration_points.shape[0] != nb_element*NbIntegrationPoints:
                raise ValueError("wrong dimensions of value_integration_points after the reshape, control the getFieldDimension() !")        

        support.fem.integrate(value_integration_points,result_integration,int_dim, support.elem_type)

        return result_integration

class Assembly:

    @staticmethod
    def assembleNodalFieldIntegration(result_integration):

        return np.sum(result_integration,axis=0)
    
    def assemblyK(groupedKlocal, support, field_dim):
        
        dim = support.fem.getMesh().getNbNodes() * field_dim
        conn = support.fem.getMesh().getConnectivity(support.elem_type)

        n_elem  = conn.shape[0]
        n_nodes_per_elem = conn.shape[1]
        numEq = np.zeros((n_elem, field_dim*n_nodes_per_elem), dtype=int)

        groupedKlocal = groupedKlocal.reshape((n_elem, 1, n_nodes_per_elem*field_dim, n_nodes_per_elem*field_dim))

        for e in range(n_elem):
            for i in range(n_nodes_per_elem):
                    for j in range(field_dim):
                        numEq[e, field_dim*i+j] = field_dim*conn[e, i]+j

        K = np.zeros((dim, dim))

        for e in range(n_elem):

            ddl = numEq[e, :]

            K_locale = groupedKlocal

            for i, gi in enumerate(ddl):
                for j, gj in enumerate(ddl):
                    K[gi, gj] += K_locale[e,0,i, j]
        return K

    def assemblyV(groupedV, support, field_dim):
        
        dim = support.fem.getMesh().getNbNodes() * field_dim
        conn = support.fem.getMesh().getConnectivity(support.elem_type)

        n_elem  = conn.shape[0]
        n_nodes_per_elem = conn.shape[1]
        numEq = np.zeros((n_elem, field_dim*n_nodes_per_elem), dtype=int)
        
        groupedV = groupedV.reshape((n_elem, 1, -1, n_nodes_per_elem*field_dim))

        for e in range(n_elem):
            for i in range(n_nodes_per_elem):
                    for j in range(field_dim):
                        numEq[e, field_dim*i+j] = field_dim*conn[e, i]+j

        V = np.zeros((field_dim, dim))

        for e in range(n_elem):

            ddl = numEq[e, :]

            V_locale = groupedV

            for i, gi in enumerate(ddl):
                    V[:, gi] += V_locale[e,0,:,i]

        return V
    
    def assemblyB(groupedV, support, field_dim):

        dim = support.fem.getMesh().getNbNodes() * field_dim
        conn = support.fem.getMesh().getConnectivity(support.elem_type)

        n_elem  = conn.shape[0]
        n_nodes_per_elem = conn.shape[1]
        numEq = np.zeros((n_elem, field_dim*n_nodes_per_elem), dtype=int)
        
        groupedV = groupedV.reshape((n_elem, 1, -1, n_nodes_per_elem*field_dim))

        for e in range(n_elem):
            for i in range(n_nodes_per_elem):
                    for j in range(field_dim):
                        numEq[e, field_dim*i+j] = field_dim*conn[e, i]+j

        if support.spatial_dimension == 1:
            V = np.zeros((field_dim, dim))
        elif support.spatial_dimension == 2 :
            V=np.zeros((3,dim))

        for e in range(n_elem):

            ddl = numEq[e, :]

            V_locale = groupedV

            for i, gi in enumerate(ddl):
                    V[:, gi] += V_locale[e,0,:,i]

        if support.spatial_dimension ==2 and field_dim==1:
            V[-1,:]=V[-1,:]*(1/2)#problème
        return V