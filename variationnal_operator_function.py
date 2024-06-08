import akantu as aka
import numpy as np

class Support:
    """
    Support class.
    
    Attributes:
        elem_filter: akaArray.
        fem: FEEngine object.
        spatial_dimension: int.
        elem_type: aka element type.
    """
    def __init__(self, elem_filter, fem, spatial_dimension, elem_type):
        self.elem_filter = elem_filter
        self.fem = fem
        self.spatial_dimension = spatial_dimension
        self.elem_type = elem_type


class TensorField:
    """
    Tensor Field class.

    Attributes:
        name: field name.
        support: Support object.
    
    Methods:
        evalOnQuadraturePoints(): estimates the field values at the quadrature points relatively to the support.
        getFieldDimension(): get dimension to be able to integrate using the FEEngine method : "fem.integrate()".
    
    Operators overloaded:
        +, -, *, @.
    """
    def __init__(self, name, support):
        self.name = name
        self.support = support

    def evalOnQuadraturePoints(self):
        """
        Estimates the field values at the quadrature points relatively to the support.

        Parameters:
            None.

        Returns:
            numpy array.
        """
        pass
    
    def getFieldDimension(self):
        """
        Get dimension to be able to integrate using the FEEngine method : "fem.integrate()".

        Parameters:
            None.

        Returns:
            integer.
        """
        pass

    def __mul__(self, f):
        """
        Overload of * Operator
        """
        return Multiplication(self, f)
    
    def __rmul__(self,f):
        """
        Overload of * Operator
        """
        return self.__mul__(f)
    
    def __add__(self, f):
        """
        Overload of + Operator
        """
        return Addition(self, f)
    
    def __radd__(self, f):
        """
        Overload of + Operator
        """
        return self.__add__(f)
    
    def __sub__(self, f):
        """
        Overload of - Operator
        """
        return Substraction(self, f)
    
    def __rsub__(self, f):
        """
        Overload of - Operator
        """
        return self.__sub__(f)
    
    def __matmul__(self, f):
        """
        Overload of @ Operator
        """
        return Contraction(self, f)
 
    
class Operator(TensorField):
    """
    Parent class of the overloaded operators and the Grandient, transpose, Curl.
    
    Attributes:
        args: list of TensorField Object.
    """
    def __init__(self, *args):
        self.args = args
        self.support = args[0].support
    
    def getFieldDimension(self):
        # True for operations between NodalTensorField, 
        # Other cases : override in derived classes
        return self.args[0].getFieldDimension()


class Addition(Operator):
    """
    Object returned by operator "+" between 2 TensorField.
    """
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
    """
    Object returned by transposing last 2 dimension of a TensorField.
    """
    def __init__(self,f):

        if not isinstance(f, (Grad, N, CurlOperator)):
            #can't be used for NodalTensorField
            raise TypeError("Be careful if you want to transpose an object different from grad(N), RotationalOp or N. It transposes the last 2 dimensions of an array. Other possibility : use Contraction class with particular subscripts: it uses einsum form numpy.")
        
        super().__init__(f)
        self.name = "transpose"+"("+f.name+")"
        self.value_integration_points = np.transpose(f.value_integration_points, axes=list(range(len(f.value_integration_points.shape) - 2)) + [-1, -2])    
    
    def evalOnQuadraturePoints(self):

        firstevaluated = self.args[0].evalOnQuadraturePoints()

        self.value_integration_points = np.transpose(firstevaluated, axes=list(range(len(firstevaluated.shape) - 2)) + [-1, -2])        
        return self.value_integration_points

    def getFieldDimension(self):
        #not meant to be integrated "raw":
        #goal of transpose: integrate an object returned by operations between multiple objects, one of which is transposed
        #therefore, to be integrate it uses the getFieldDimension of the contraction (for example BtDB) but never this one directly.
        raise NotImplementedError("Not possible to directly integrate a transpose.")

class Substraction(Operator):
    """
    Object returned by operator "-" between 2 TensorField.
    """
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
    """
    Object returned by operator "*" between 2 TensorField.
    """
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
    """
    NodalTensorField class : instanciate using values at each node (a nodal_vector).
    """
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
    """
    GenericOperator : use einsum from numpy. Have exactly the same application.
    
    Attributes:
        subscripts_for_summation: contains the indices of the operation.
    
    Operator overload:
        () : specifies on which NodalTensorField the operation is done.
    """
    def __init__(self, *args, final=None):
        """
        Parameters:
            *args : one string per TensorField, each string contains the dimension of its Tensorfield. See numpy.einsum ref : part before "->".
            final : indice of the result TensorField. Set on which indice the einsum is done. see numpy.einsum ref : part after "->".
        """
        if final==None:
            raise ValueError("please give the indices of result array")
        self.subscripts_for_summation = ','.join(["xy"+i for i in args])
        self.subscripts_for_summation+= "->" + "xy" + final

    def __call__(self,*args):
        """
        Overload of ().
        """
        return Contraction(*((self.subscripts_for_summation,) + args)) #return a Contraction Object even if it's not a real contraction. But works like this.


class Contraction(Operator):
    """
    Object returned by operator @ between 2 TensorField or by () of a GenericOperator object.
        If @ : Contract last 2 dimensions. (used with only 2 TensorField).
        If () : Do the eisum with respect to the "subcript" of the GenericOperator Object.
    """
    def __init__(self, *args):

        if isinstance(args[0], str):
            self.subscripts_for_summation = args[0]
            super().__init__(*args[1:])
        else :
            self.subscripts_for_summation = None
            super().__init__(*args)

        self.value_integration_points = None 

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
    """
    ShapeField class.
    
    Member :
        value_integration_points :
        Contains exactly the output of getShapes of the FEEngine of Akantu.
        It contains the Shape function evaluated on the quadrature point. It has the shape (nb_element * nb_integration_point_per_element,nb_nodes_per_elem).
    """
    def __init__(self, support, dim_field):#dim_field in parameter. Can maybe be improved. Not used in this class but necessary if we want to write directly grad(...) without giving the dim.
        super().__init__("shape_function", support)
        self.dim_field = dim_field
        self.NbIntegrationPoints=support.fem.getNbIntegrationPoints(support.elem_type)
        self.conn = support.fem.getMesh().getConnectivities()(support.elem_type)
        self.nb_elem = self.conn.shape[0]
        self.nb_nodes_per_elem = self.conn.shape[1]

        self.value_integration_points = np.zeros((self.nb_elem * self.NbIntegrationPoints, self.nb_nodes_per_elem))
    
    def evalOnQuadraturePoints(self):

        self.value_integration_points = self.support.fem.getShapes(self.support.elem_type)
        return self.value_integration_points
    
    def getFieldDimension(self):
        # Never directly integrate
        raise NotImplementedError
    
class N(ShapeField):
    """
    N class.
    
    Member :
        value_integration_points :
        Matrix that contain the shape function evaluated in the quad point.
        The shape function are at the right position in "N" to have : u = N @ nodal_vector.
        It has the shape (nb_element, nb_integration_point_per_element, field_dimension, nb_nodes_per_elem * field_dimension).
    """
    def __init__(self, support, dim_field):
        super().__init__(support,dim_field)
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

class ConstitutiveLaw(ShapeField):
    """
    Constitutive class : this class was necessary to do the patch test for Navier eq. But it's temporary.
    """
    def __init__(self, D, support): 
        super().__init__(support, dim_field = None)
        self.D = D

        if self.support.spatial_dimension == 2 :
            self.value_integration_points = np.zeros((self.nb_elem,1,3,3))
        
        elif self.support.spatial_dimension == 1 :
            #implémenter uniquement pour le patch test avec trianglre donc pas de cas 1 D
            raise NotImplementedError

    def evalOnQuadraturePoints(self):
        D = self.D.reshape((1,1,3,3))
        self.value_integration_points = np.tile(D, (self.nb_elem, 1, 1, 1))
        return self.value_integration_points
    
class Grad(Operator):
    """
    Gradient operator class.

    Differential Operator implemented for :
        Grad(ShapeField Object), give a "real" gradient. It was used for Heat equation.
        Grad(N Object), give the matrix B such as epsilon = B@nodal_vector. It was used for Navier eq.

    Value_integration_points :
        if Grad(ShapeField Object) has the shape (nb_element,nb_integration_point_per_element, spatial_dimension, nb_nodes_per_elem*field_dimension).
        if Grad(N Object) has the shape (nb_element, nb_integration_point_per_element, 1(1D) or 3(2D), nb_nodes_per_elem*spatial_dimension ).
    """
    def __init__(self, f1):
        super().__init__(f1)
        self.name = "Gradient(" + f1.name + ")"

        self.conn = f1.conn
        self.nb_elem = self.conn.shape[0]
        self.NbIntegrationPoints=f1.NbIntegrationPoints
        self.nb_nodes_per_elem = self.conn.shape[1]

        if isinstance(f1, N):
            
            dim_field = f1.dim_field

            if self.support.spatial_dimension == 1 :
                nb_line = 1

            elif self.support.spatial_dimension == 2 :
                nb_line = 3

            self.value_integration_points = np.zeros((self.nb_elem, self.NbIntegrationPoints, nb_line, self.nb_nodes_per_elem *dim_field))
        
        elif isinstance(f1, ShapeField):
            dim_field = f1.dim_field
            self.value_integration_points = np.zeros((self.nb_elem, self.NbIntegrationPoints, self.support.spatial_dimension, self.nb_nodes_per_elem * dim_field))

        else : 
            raise NotImplementedError("gradient is implemented only for a shapefield")
          
    
    def evalOnQuadraturePoints(self):

        if isinstance(self.args[0], N):
            
            dim_field = self.args[0].dim_field
            B_without_dim_extension = np.zeros((self.nb_elem, self.NbIntegrationPoints,1, self.nb_nodes_per_elem*self.support.spatial_dimension))
            derivatives_shapes = self.support.fem.getShapesDerivatives(self.support.elem_type)
            derivatives_shapes = derivatives_shapes.reshape((B_without_dim_extension.shape))

            if self.support.spatial_dimension == 1:
                for i in range(dim_field):
                    self.value_integration_points[:,:,i::dim_field,i::dim_field]=derivatives_shapes

            elif self.support.spatial_dimension == 2:      
                if dim_field == 2:
                    self.value_integration_points[:,:,0,0::dim_field]=derivatives_shapes[:,:,0,::self.support.spatial_dimension]
                    self.value_integration_points[:,:,1,1::dim_field]=derivatives_shapes[:,:,0,1::self.support.spatial_dimension]
                    self.value_integration_points[:,:,2,1::dim_field]=derivatives_shapes[:,:,0,::self.support.spatial_dimension]
                    self.value_integration_points[:,:,2,0::dim_field]=derivatives_shapes[:,:,0,1::self.support.spatial_dimension]
        
        elif isinstance(self.args[0], ShapeField):
            dim_field = self.args[0].dim_field
            if dim_field == 1 :
                B_without_dim_extension = np.zeros((self.nb_elem, self.NbIntegrationPoints,1,self.support.spatial_dimension * self.nb_nodes_per_elem))
                derivatives_shapes = self.support.fem.getShapesDerivatives(self.support.elem_type)
                derivatives_shapes = derivatives_shapes.reshape((B_without_dim_extension.shape))
                
                for i in range(self.support.spatial_dimension):
                        self.value_integration_points[:,:,i,:]=derivatives_shapes[:,:,0,i::self.support.spatial_dimension]
            else :
                raise NotImplementedError("Only field dimension = 1 has been considered")
            
        return self.value_integration_points
    
    def getFieldDimension(self):

        return np.prod(self.value_integration_points.shape[-2:])
    
class CurlOperator(Operator):
    """
    CurlOperator class. Currently only implemented for a 3D case.

    Differential operator implemented for :
        CurlOperator(ShapeField Object).
    
    Value_integration_points (remind : only in 3D):
        has the shape (nb_element,nb_integration_point_per_element, 3, nb_nodes_per_elem*3).

    """
    def __init__(self, f1):
        super().__init__(f1)
        self.name = "Rot(" + f1.name + ")"
        dim_field = f1.dim_field
        self.conn = f1.conn
        self.nb_elem = self.conn.shape[0]
        self.NbIntegrationPoints = f1.NbIntegrationPoints
        self.nb_nodes_per_elem = self.conn.shape[1]

        if dim_field !=3:
            raise NotImplementedError("curl operator is implemented only for a 3D case !")

        if isinstance(f1, ShapeField):

            self.value_integration_points = np.zeros((self.nb_elem, self.NbIntegrationPoints, self.support.spatial_dimension, self.nb_nodes_per_elem * dim_field))

        else : 
            raise NotImplementedError("curl operator is implemented only for a shapefield")
          
    
    def evalOnQuadraturePoints(self):
        dim_field = self.args[0].dim_field
        B_without_dim_extension = np.zeros((self.nb_elem, self.NbIntegrationPoints,1,self.support.spatial_dimension * self.nb_nodes_per_elem))
        derivatives_shapes = self.support.fem.getShapesDerivatives(self.support.elem_type)
        derivatives_shapes = derivatives_shapes.reshape((B_without_dim_extension.shape))


        if self.support.spatial_dimension == 3:
            #line 1   
            self.value_integration_points[:,:,0,2::dim_field]=derivatives_shapes[:,:,0,1::self.support.spatial_dimension]
            self.value_integration_points[:,:,0,1::dim_field]=(-1)*derivatives_shapes[:,:,0,2::self.support.spatial_dimension]
            #line 2
            self.value_integration_points[:,:,1,0::dim_field]=derivatives_shapes[:,:,0,2::self.support.spatial_dimension]
            self.value_integration_points[:,:,1,2::dim_field]=(-1)*derivatives_shapes[:,:,0,0::self.support.spatial_dimension]
            #line 3
            self.value_integration_points[:,:,2,1::dim_field]=derivatives_shapes[:,:,0,0::self.support.spatial_dimension]
            self.value_integration_points[:,:,2,0::dim_field]=(-1)*derivatives_shapes[:,:,0,1::self.support.spatial_dimension]
        else :
            raise NotImplementedError("curl operator is implemented only for a 3D case !")
        
        return self.value_integration_points
        
    
    def getFieldDimension(self):

        raise np.prod(self.value_integration_points.shape[-2:])
    
class FieldIntegrator:
    """
    FieldIntegrator class (static class).

    Methods:
        integrate : integrate a TensorField.
    """
    @staticmethod
    def integrate(field):
        """
        Integrate a TensorField.

        Parameters:
            TensorField

        Returns:
            numpy array of shape "(nb_element,-1)".
        """
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
    """
    Assembly class (static class).
    
    Methods:
        assembleNodalFieldIntegration : Assemble the result of "integrate" for a NodalTensorField.
        assemblyK : Assemble a stiffness matrix (case Navier eq) or thermal conductivity matrix (heat eq.) or ... (rotationnal case) after "integrate". 
        assemblyV : Assemble the integration of N Object or Grad(ShapeField object). !!! test case Grad(ShapeField object)!!!
        assemblyB : Assemble the integration of grad(N) Object.
    """
    @staticmethod
    def assembleNodalFieldIntegration(result_integration):
        """
        Assembled the result of "integrate" for a NodalTensorField.

        Parameters:
            array : result of the integrate method.

        Returns:
            numpy array of shape "(nb_element,-1)".
        """

        return np.sum(result_integration,axis=0)
    
    def assemblyK(groupedKlocal, support, field_dim):
        """
        Assemble a stiffness matrix (case Navier eq) or thermal conductivity matrix (heat eq.) or ... (rotationnal case) after "integrate". 

        Parameters:
            groupedKlocal : numpy array, result of the "integrate".
            support : Support Object.
            field_dim : field dimension.

        Returns:
            numpy array of shape "(nb_nodes*field_dimension,nb_nodes*field_dimension)".
        """
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
        """
        Assemble the integration of N Object or Grad(ShapeField object). !!! test case Grad(ShapeField object)!!!
        
        Parameters:
            groupedV : numpy array, result of the "integrate".
            support : Support Object.
            field_dim : field dimension.

        Returns:
            numpy array of shape "(field_dimension,nb_nodes*field_dimension)".
        """
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
        """
        Assemble the integration of grad(N) Object. 

        Parameters:
            groupedV : numpy array, result of the "integrate".
            support : Support Object.
            field_dim : field dimension.

        Returns:
            numpy array.
        """
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

        return V