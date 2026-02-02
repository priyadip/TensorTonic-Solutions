import numpy as np

def apply_homogeneous_transform(T, points):
    single_point = False

    # Ensure numpy arrays
    points = np.asarray(points)

    # Handle single point
    if points.ndim == 1:
        points = points.reshape(1, 3)
        single_point = True

    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))   # (N, 4)

    # Apply transformation
    transformed_h = (T @ points_h.T).T     # (N, 4)

    # Extract spatial coordinates
    result = transformed_h[:, :3]

    # Restore original shape
    if single_point:
        return result.reshape(3,)

    return result
