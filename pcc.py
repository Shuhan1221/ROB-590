import numpy as np

def compute_rotation_matrix(phi, beta):
    """
    Compute the rotation matrix given phi and beta angles.
    
    Parameters:
    phi : float - Directional angle (radians)
    beta : float - Bending angle (radians)
    
    Returns:
    3x3 rotation matrix (numpy array)
    """
    a = np.cos(beta) * np.cos(phi)**2 + np.sin(phi)**2
    b = (-1 + np.cos(beta)) * np.cos(phi) * np.sin(phi)
    c = np.cos(phi) * np.sin(beta)
    d = np.sin(beta) * np.sin(phi)
    e = np.cos(beta)
    f = np.cos(beta) * np.sin(phi)**2 + np.cos(phi)**2

    return np.array([[a, b, c], [b, f, d], [-c, -d, e]])

def calc_displacement(rho, beta, phi):
    """
    Compute the displacement vector based on curvature parameters.
    
    Parameters:
    rho : float - Curvature
    beta : float - Bending angle (radians)
    phi : float - Directional angle (radians)
    
    Returns:
    3D displacement vector (numpy array)
    """
    return (1 / rho) * np.array([(1 - np.cos(beta)) * np.sin(phi),
                                 (1 - np.cos(beta)) * np.cos(phi),
                                 np.sin(beta)])

def get_beta(l1, l2, l3, r):
    """
    Compute the bending angle beta from segment lengths.
    
    Parameters:
    l1, l2, l3 : float - Three segment lengths
    r : float - Radius parameter
    
    Returns:
    Beta (float) - Bending angle in radians
    """
    return 2 * np.sqrt(l1**2 + l2**2 + l3**2 - l1*l2 - l1*l3 - l2*l3) / (3 * r)

def get_phi(l1, l2, l3):
    """
    Compute the directional angle phi from segment lengths.
    
    Parameters:
    l1, l2, l3 : float - Three segment lengths
    
    Returns:
    Phi (float) - Directional angle in radians
    """
    return np.arctan2(3 * (l2 - l3), np.sqrt(3) * (l2 + l3 - 2 * l1))

def length_to_position_single(l, rigid_len, r):
    """
    Compute the end-effector position for a single segment.
    
    Parameters:
    l : list - Three segment lengths [l1, l2, l3]
    rigid_len : float - Length of the rigid body
    r : float - Radius parameter
    
    Returns:
    3D end-effector position (numpy array)
    """
    # Compute parameters
    phi = get_phi(l[0], l[1], l[2])
    beta = get_beta(l[0], l[1], l[2], r)
    lc = np.mean(l)  # Average segment length
    rho = beta / lc  # Curvature

    # Compute displacement and rotation
    displacement = calc_displacement(rho, beta, phi)
    rotation = compute_rotation_matrix(phi, beta)
    direction = rotation @ np.array([0, 0, 1])  # Transform the direction

    # Compute final position
    end_position = displacement + direction * rigid_len
    #return end_position_final
    return end_position


# Example usage with arbitrary actuator lengths
l1, l2, l3 = 60, 80, 100  # Example lengths
final_position = compute_final_position(l1, l2, l3)
final_position
