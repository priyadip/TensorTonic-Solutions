import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    # Initialize indices 0..N-1
    indices = np.arange(N)
    
    # Use provided random generator or default to numpy's default
    if rng is None:
        rng = np.random.default_rng()
    
    # Shuffle indices if required for cross-validation
    if shuffle:
        rng.shuffle(indices)
    
    # Determine fold sizes. Each fold should differ in size by at most 1
    fold_size = N // k
    remainder = N % k
    
    folds = []
    current = 0
    for i in range(k):
        # Distribute the remainder across the first few folds
        size = fold_size + (1 if i < remainder else 0)
        folds.append(indices[current:current + size])
        current += size
        
    # Create (train, val) pairs for each fold i
    results = []
    for i in range(k):
        val_idx = folds[i]
        # Training indices are the union of all other folds
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        # Ensure indices are returned in a standard format (e.g., numpy arrays)
        results.append((train_idx, val_idx))
        
    return results