import numpy as np

def calc_laplacian_from_conductance_matrix(C):
    if not np.allclose(C, C.T):
        raise ValueError("Conductance matrix must be symmetric")
    if not np.all(np.diagonal(C) == 0):
        raise ValueError("Conductance matrix diagonal must be zero")
    diag = np.sum(C, axis=1)
    L = np.diag(diag) - C
    return L

def calc_two_point_from_laplacian_matrix(L, node_idx1, node_idx2):
    """ This function calculates the two point resistance between two nodes in a network given the Laplacian matrix.

    Args:
        L (np.array): Laplacian matrix of the network (hermitian)
        node_idx1 (int): idx of node 1 
        node_idx2 (int): idx of node 2
                         Note idxs start at zero, so the idxs are 1 lower than in the Wu paper
    """
    
    # calculate the eigenvalues and orthonormal eigenvectors of the Laplacian matrix
    eigvals, eigvecs = np.linalg.eigh(L)

    # calculate the effective resistance between the two nodes
    R = 0
    for II in range(1, len(eigvals)):
        if eigvals[II] > 1e-10:
            R += (1/eigvals[II]) * (eigvecs[node_idx1, II] - eigvecs[node_idx2, II])**2

    return R

##### Here is where you create your conductance and laplacian matrices then calc the resistance.
##### see the examples below for how to do this




#####


DO_EXAMPLES = True # Set this to False when you are running your own code above, the below are
                     # examples from the Wu et al. paper

if DO_EXAMPLES:
    
    # Example 1 4-node network from Wu paper:
    # resistances 
    r_1 = 1
    r_2 = 3

    # conductance matrix
    c_1 = 1/r_1
    c_2 = 1/r_2
    C = np.array([[0, c_1, 0, c_1],
                [c_1, 0, c_1, c_2],
                [0, c_1, 0, c_1],
                [c_1, c_2, c_1, 0]])
    # laplacian matrix
    L = calc_laplacian_from_conductance_matrix(C)


    # Note idxs start at zero, so the idxs are 1 lower than in the Wu paper
    R02_check = r_1 # from Wu paper
    R02 = calc_two_point_from_laplacian_matrix(L, 0, 2)
    print(R02, R02_check)

    R01_check = r_1*(2*r_1 + 3*r_2)/(4*(r_1 + r_2)) # from Wu paper
    R01 = calc_two_point_from_laplacian_matrix(L, 0, 1)
    print(R01, R01_check)

    if (np.abs(R01 - R01_check) < 1e-8) and (np.abs(R02 - R02_check) < 1e-8):
        print("Test for Example 1 from Wu et al. passed")
    else:
        print("WARNING: Test for Example 1 from Wu et al. failed")

    # Example 2 complete graph network with N nodes

    # number of nodes
    N = 4
    r = 2
    L = -1*np.ones((N, N))
    np.fill_diagonal(L, N-1)
    L = r**-1*L

    R01_N4_check = 2*r/(N)
    R01_N4 = calc_two_point_from_laplacian_matrix(L, 0, 1)
    print(R01_N4, R01_N4_check)

    N = 11
    r = 3
    # for this example I assign L directly, see the other examples for calculating L from C, the conduction matrix
    L = -1*np.ones((N, N))
    np.fill_diagonal(L, (N-1))
    L = r**-1*L

    R01_N11_check = 2*r/(N)
    R01_N11 = calc_two_point_from_laplacian_matrix(L, 0, 1)
    print(R01_N11, R01_N11_check)

    if (np.abs(R01_N4 - R01_N4_check) < 1e-8) and (np.abs(R01_N11 - R01_N11_check)) < 1e-8:
        print("Test for Example 2 passed")
    else:
        print("WARNING: Test for Example 2 failed")

        
    # Section 3.1 one-dimensional lattice with N nodes, free boundaryies

    N = 4
    r = 0.5
    C = np.zeros((N, N))
    for II in range(N-1):
        C[II, II+1] = 1/r
        C[II+1, II] = 1/r
    L = calc_laplacian_from_conductance_matrix(C)

    R03_N4_check = r*(3 - 0)
    R03_N4 = calc_two_point_from_laplacian_matrix(L, 0, 3)
    print(R03_N4, R03_N4_check)

    if (np.abs(R03_N4 - R03_N4_check) < 1e-8):
        print("Test for section 3.1 passed")
    else:
        print("WARNING: Test for section 3.1 failed")

    # Section 3.2 one-dimensional lattice with N nodes, periodic boundaries

    N = 8
    r = 0.33
    C = np.zeros((N, N))
    for II in range(N-1):
        C[II, II+1] = 1/r
        C[II+1, II] = 1/r
    C[0, N-1] = 1/r
    C[N-1, 0] = 1/r

    L = calc_laplacian_from_conductance_matrix(C)

    node_idx1 = 1
    node_idx2 = 5
    R15_N8_check = r*np.abs(node_idx1 - node_idx2)*(1 - np.abs(node_idx1 - node_idx2)/N)
    R15_N8 = calc_two_point_from_laplacian_matrix(L, node_idx1, node_idx2)
    print(R15_N8, R15_N8_check)

    if (np.abs(R15_N8 - R15_N8_check) < 1e-8):
        print("Test for section 3.2 passed")
    else:
        print("WARNING: Test for section 3.2 failed")

        
    # Example 3 2D square lattice with N nodes
    # NOTE: The diagram in the Wu et al. paper is wrong, they have a 6x4 network but the text says 5x4

    M = 5
    N = 4
    r = 0.33

    C = np.zeros((M*N, M*N))
    for II in range(M*N):
        if (II % M) != (M-1):
            C[II, II+1] = 1/r
            C[II+1, II] = 1/r
        if (II + M) < M*N:
            C[II, II+M] = 1/r
            C[II+M, II] = 1/r
    # print(C)

    L = calc_laplacian_from_conductance_matrix(C)

    node_idx1 = 0 # (0,0)
    node_idx2 = M*3 + 3  # (3,3)
    R03_N20_check = (3/4 + 3/5 + 9877231/27600540)*r
    R03_N20 = calc_two_point_from_laplacian_matrix(L, node_idx1, node_idx2)
    print(R03_N20, R03_N20_check)

    if (np.abs(R03_N20 - R03_N20_check) < 1e-8):
        print("Test for Example 3 passed")
    else:
        print("WARNING: Test for Example 3 failed")






