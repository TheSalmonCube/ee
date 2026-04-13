import numpy as np
from scipy import sparse

def lanczos(A: sparse.csr_matrix, v0: np.ndarray, m: int):
    n = v0.shape[0]

    alphas = np.zeros(m, dtype=np.complex64)
    betas = np.zeros(m, dtype=np.complex64)
    Q = np.zeros((n, m), dtype=np.complex64)

    # normalize v
    v = v0 / np.linalg.norm(v0)
    Q[:, 0] = v
    
    for i in range(m):   
        # apply A to latest v for the new direction     
        w = A @ v

        # orthogonalize against all previous v (see note 1)
        for j in range(i):
            w -= np.vdot(Q[:, j], w) * Q[:, j]
        
        # alpha is the projection of A*v on v
        alphas[i] = np.vdot(v, w)
        v = w - alphas[i] * v

        # beta is the norm of the new v
        betas[i] = np.linalg.norm(v)
        v = v / betas[i]

        # store the next vector in Q unless we are at the final step
        if i < m - 1:
            Q[:, i+1] = v

    # the final beta is just the norm of the (m+1)th vector, unneccessary
    return Q, alphas, betas[:-1]

# note 1 - The true lanczos algorithm only orthogonalizes against the previous two terms
# in a three term reccurance, enough for a hermitian A. I expect hermitian A, but with numerical
# limitations, the three term reccurance does not make it exactly orthogonal. 
# This implementation uses full orthogonalization, which is more expensive but more accurate.
# Lanczos is O(nm) this is O(nmm). As long as m is not large it should be fine, as m << n.

### DO NOT USE (SLOW) FOR REFERENCE ONLY
### Highly advised to use np.linalg.eig(A).

def tridiagonaleigen(diagonal, offdiagonal):
    m = diagonal.shape[0]
    V = np.identity(m) # eigenvectors
    p = 0
    k = m-1
    max_iterations = 30

    while k > p:
        for iter in range(max_iterations):
            # 0. REMOVE INITIAL EIGENVALUES (first offdiagonal is zero)
            while offdiagonal[p] == 0:
                p += 1

            # 1. SHIFT: closed eigen eigenvalue to a_K of
            # [ a_k-1  b_k-1 ]
            # [ b_k-1  a_k   ]
            # eigenvalue computed with the characteristic quadratic
            shift = (diagonal[k] + diagonal[k-1]) / 2 + np.sign(diagonal[k] - diagonal[k-1]) * np.sqrt(offdiagonal[k-1]**2 + 0.25*(diagonal[k] - diagonal[k-1])**2)
            
            # 2. FIRST GIVENS ROTATION: construct G1 by finding the rotation matrix G1 = 
            # [ cos  sin ]
            # [-sin  cos ]
            # such that G1 * [ a1-shift, b1 ] = [ r, 0 ] for some r.
            a1 = diagonal[p]
            a2 = diagonal[p+1]
            b1 = offdiagonal[p]
            b2 = offdiagonal[p+1]

            r = np.hypot(a1-shift, b1) # since rotations are norm preserving
            if r == 0:
                cos = 1.0
                sin = 0.0
            else:
                cos = (a1-shift)/r 
                sin = b1/r #trig

            # apply: T = G1 T G1^T. trust that the matmul expands to this
            diagonal[p]   = cos**2 * a1 + 2 * cos * sin * b1 + sin**2 * a2
            diagonal[p+1] = sin**2 * a1 - 2 * cos * sin * b1 + cos**2 * a2
            offdiagonal[p] = (cos**2 - sin**2) * b1 + cos * sin * (a2 - a1)
            offdiagonal[p+1] = cos * b2
            bulge = sin * b2 # this is an element not on the tridiagonals at [p, p+2] in T

            # apply G1 to V to keep track of eigenvectors
            v1 = V[:, p].copy()
            v2 = V[:, p+1].copy()
            V[:, p]   = cos * v1 + sin * v2
            V[:, p+1] = -sin * v1 + cos * v2

            # 3. BULGE CHASE (more givens rotations):
            for i in range(p, k - 1):
                # construct Gi+1 by finding the rotation matrix such that
                # Gi+1 * [ bi, bulge ] = [ r, 0 ].
                # where bi is at (i, i+1) and bulge is at (i, i+2)
                
                bi = offdiagonal[i]
                ai1 = diagonal[i+1]
                ai2 = diagonal[i+2]
                bi1 = offdiagonal[i+1]
                
                r = np.hypot(bi, bulge)
                if r == 0:
                    cos = 1.0
                    sin = 0.0
                else:
                    cos = bi/r
                    sin = bulge/r

                # apply: T = G T G^T.
                offdiagonal[i] = r # as required
                diagonal[i+1] = cos**2 * ai1 + 2 * cos * sin * bi1 + sin**2 * ai2
                diagonal[i+2] = sin**2 * ai1 - 2 * cos * sin * bi1 + cos**2 * ai2
                offdiagonal[i+1] = (cos**2 - sin**2) * bi1 + cos * sin * (ai2 - ai1)

                # apply Gn to v to keep track of eigenvectors
                v1 = V[:, i+1].copy()
                v2 = V[:, i+2].copy()
                V[:, i+1] = cos * v1 + sin * v2
                V[:, i+2] = -sin * v1 + cos * v2
                
                if i + 2 < k:
                    # iff there is a next off-diagonal element, it becomes the new bulge
                    bi2 = offdiagonal[i+2]
                    offdiagonal[i+2] = cos * bi2
                    bulge = sin * bi2
                else:
                    bulge = 0
                    break
            
            # 4. CHECK CONVERGENCE:
            tolerance = 1e-15
            if np.abs(offdiagonal[k-1]) < tolerance:
                k -= 1
                break

            if iter == max_iterations-1:
                print("QR algorithm did not converge after ", max_iterations, " iterations")

    return V, diagonal

    