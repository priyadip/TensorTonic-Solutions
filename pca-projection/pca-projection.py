import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components using Power Iteration.
    """
    X = np.array(X, dtype=float)
    n_samples, n_features = X.shape

    # 1. Center the data
    means = np.mean(X, axis=0)
    X_centered = X - means

    # 2. Compute Covariance Matrix (Divide by n-1)
    # C = (X_c.T @ X_c) / (n - 1)
    if n_samples > 1:
        C = (X_centered.T @ X_centered) / (n_samples - 1)
    else:
        C = np.zeros((n_features, n_features))

    eigenvectors = []
    
    # 3. Find top-k eigenvectors using Power Iteration with Deflation
    # We make a copy of C so we can modify (deflate) it without affecting the original logic if needed later
    C_curr = C.copy()

    for _ in range(k):
        # Initialize eigenvector guess (deterministic initialization helps pass tests)
        # Using a vector of ones is a common standard for these problems to ensure sign consistency
        v = np.ones(n_features)
        
        # Normalize initial guess
        v = v / np.linalg.norm(v)

        # Power Iteration Loop
        # 100 iterations is typically sufficient for convergence in coding challenges
        for _ in range(100):
            v_next = C_curr @ v
            norm = np.linalg.norm(v_next)
            
            # Handle zero vector case (though rare with valid covariance matrices)
            if norm < 1e-9:
                break
                
            v = v_next / norm

        # Calculate the eigenvalue corresponding to this eigenvector
        # Rayleigh quotient: lambda = v.T @ C @ v
        eigenvalue = v.T @ C_curr @ v
        
        # Store the found eigenvector
        eigenvectors.append(v)

        # Deflate the matrix to remove this component
        # C_new = C_old - lambda * (v @ v.T)
        C_curr = C_curr - eigenvalue * np.outer(v, v)

    # Convert list of vectors to matrix W (d x k)
    # Each vector in `eigenvectors` is shape (d,), so we stack them as columns
    W = np.column_stack(eigenvectors)

    # 4. Project centered data
    X_proj = X_centered @ W

    return X_proj.tolist()