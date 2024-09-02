import numpy as np
from scipy import linalg as LA

def compute_A_ElasticNet(X, coef_, alpha=None, l1_ratio=None):
    """
    Compute the interaction matrix for MultitaskElasticNet

    arguments: X            shape (n,p)
               coef_        shape (n_tasks, p)
               alhpa        float, positive              
               l1_ratio     float in [0,1]

    """
    assert X.shape[1] == coef_.shape[1]
    n = X.shape[0]
    n_tasks = coef_.shape[0]

    S_hat = np.nonzero(LA.norm(coef_, axis=0))[0]
    hbB_S = coef_[:,S_hat]
    X_S = X[:,S_hat]
    p_S = len(S_hat)
    
    vector_of_inverse_euclidean_norms = LA.norm(hbB_S, axis=0)**(-1)
    b = U_fast = np.sqrt( alpha*l1_ratio ) * np.einsum('tj,j->tj',
                                                   hbB_S,
                                                   vector_of_inverse_euclidean_norms**1.5,
                                                   optimize='optimal')
    gram = X_S.T @ X_S /n 
    diag_vector = alpha*l1_ratio * vector_of_inverse_euclidean_norms
    X_T_X_plus_diag_inverse = LA.inv(gram + alpha*(1.0-l1_ratio)*np.eye(p_S) + np.diag(diag_vector))
    
    A_first_part = np.eye(n_tasks) * np.trace( gram @ X_T_X_plus_diag_inverse )

    second_matrix = LA.inv(-np.eye(p_S) + np.einsum("tj, tk, jk->jk", b, b, X_T_X_plus_diag_inverse))
    
    A_second_part = np.einsum("tj, sk, jk, jk -> ts",
                              b, b, X_T_X_plus_diag_inverse @ gram @ X_T_X_plus_diag_inverse,
                              second_matrix, optimize='optimal')
    A = A_first_part - A_second_part
    return A
