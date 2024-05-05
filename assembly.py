import numpy as np

def assemblyK(conn, groupedKlocal, dim1, dim2, field_dim):

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