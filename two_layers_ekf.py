import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------- PCC Forward Kinematics Functions ----------------

def compute_rotation_matrix(phi, beta):
    """
    Compute the rotation matrix given directional angle phi and bending angle beta.
    
    Parameters:
      phi  : float - Directional angle (radians)
      beta : float - Bending angle (radians)
    
    Returns:
      3x3 rotation matrix (numpy array)
    """
    a = np.cos(beta) * np.cos(phi)**2 + np.sin(phi)**2
    b = (-1 + np.cos(beta)) * np.cos(phi) * np.sin(phi)
    c = np.cos(phi) * np.sin(beta)
    d = np.sin(beta) * np.sin(phi)
    e = np.cos(beta)
    f_val = np.cos(beta) * np.sin(phi)**2 + np.cos(phi)**2
    return np.array([[a, b, c],
                     [b, f_val, d],
                     [-c, -d, e]])

def calc_displacement(rho, beta, phi):
    """
    Compute the displacement vector based on curvature parameters.
    
    Parameters:
      rho  : float - Curvature
      beta : float - Bending angle (radians)
      phi  : float - Directional angle (radians)
    
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
      r          : float - Radius parameter
      
    Returns:
      beta (float) in radians
    """
    return 2 * np.sqrt(l1**2 + l2**2 + l3**2 - l1*l2 - l1*l3 - l2*l3) / (3 * r)

def get_phi(l1, l2, l3):
    """
    Compute the directional angle phi from segment lengths.
    
    Parameters:
      l1, l2, l3 : float - Three segment lengths
      
    Returns:
      phi (float) in radians
    """
    return np.arctan2(3 * (l2 - l3), np.sqrt(3) * (l2 + l3 - 2 * l1))

def pcc_transform(u, rigid_len, r):
    """
    Compute the homogeneous transformation (4x4 matrix) for one continuum segment.
    
    Parameters:
      u         : numpy array - Three segment lengths [l1, l2, l3]
      rigid_len : float - Rigid extension length of the segment
      r         : float - Radius parameter for the PCC model
      
    Returns:
      T : 4x4 homogeneous transformation matrix representing the segment's pose.
    """
    phi = get_phi(u[0], u[1], u[2])
    beta = get_beta(u[0], u[1], u[2], r)
    lc = np.mean(u)      # average segment length
    rho = beta / lc      # curvature
    disp = calc_displacement(rho, beta, phi)
    R_seg = compute_rotation_matrix(phi, beta)
    # Rigid extension along the local z-axis (transformed by R_seg)
    direction = R_seg @ np.array([0, 0, 1])
    p = disp + direction * rigid_len
    T = np.eye(4)
    T[0:3, 0:3] = R_seg
    T[0:3, 3] = p
    return T

def rotmat_to_euler(R):
    """
    Convert a rotation matrix R (3x3) to Euler angles (roll, pitch, yaw)
    using the ZYX convention.
    
    Parameters:
      R : 3x3 numpy array
      
    Returns:
      euler : numpy array [roll, pitch, yaw] (in radians)
    """
    if abs(R[2,0]) < 1:
        pitch = -np.arcsin(R[2,0])
        cos_pitch = np.cos(pitch)
        roll = np.arctan2(R[2,1] / cos_pitch, R[2,2] / cos_pitch)
        yaw = np.arctan2(R[1,0] / cos_pitch, R[0,0] / cos_pitch)
    else:
        yaw = 0
        if R[2,0] <= -1:
            pitch = np.pi/2
            roll = np.arctan2(R[0,1], R[0,2])
        else:
            pitch = -np.pi/2
            roll = np.arctan2(-R[0,1], -R[0,2])
    return np.array([roll, pitch, yaw])

def euler_to_rotmat(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a rotation matrix using ZYX convention.
    
    Parameters:
      roll, pitch, yaw : floats in radians
      
    Returns:
      R : 3x3 rotation matrix (numpy array)
    """
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    return Rz @ Ry @ Rx

def state_to_transform(mu):
    """
    Convert a state vector [x, y, z, roll, pitch, yaw] into a homogeneous transformation matrix.
    
    Parameters:
      mu : numpy array (6,)
    
    Returns:
      T : 4x4 homogeneous transformation matrix
    """
    x, y, z = mu[0:3]
    roll, pitch, yaw = mu[3:6]
    R = euler_to_rotmat(roll, pitch, yaw)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = np.array([x, y, z])
    return T

def arc_curve(u, r, num_points=100):
    """
    Compute a set of points along the flexible (arc) portion of one continuum segment.
    
    Parameters:
      u         : numpy array - Three segment lengths [l1, l2, l3]
      r         : float - Radius parameter
      num_points: int   - Number of points along the arc
      
    Returns:
      points : numpy array of shape (num_points, 3) representing points along the arc.
    """
    phi = get_phi(u[0], u[1], u[2])
    beta = get_beta(u[0], u[1], u[2], r)
    lc = np.mean(u)
    rho = beta / lc
    t_vals = np.linspace(0, 1, num_points)
    points = []
    for t in t_vals:
        disp_t = (1 / rho) * np.array([(1 - np.cos(t * beta)) * np.sin(phi),
                                       (1 - np.cos(t * beta)) * np.cos(phi),
                                       np.sin(t * beta)])
        points.append(disp_t)
    return np.array(points)

# ---------------- EKF Update for Layer 1 ----------------

def f_layer1(u1, rigid_len, r):
    """
    Compute the predicted state for layer 1 from control input u1.
    
    Parameters:
      u1        : numpy array, control input for layer 1 [l1, l2, l3]
      rigid_len : float, rigid extension length for layer 1
      r         : float, PCC model radius parameter
      
    Returns:
      state1_pred : numpy array (6,) = [x1, y1, z1, roll1, pitch1, yaw1]
    """
    T1 = pcc_transform(u1, rigid_len, r)
    pos1 = T1[0:3, 3]
    euler1 = rotmat_to_euler(T1[0:3, 0:3])
    return np.hstack((pos1, euler1))

def ekf_update_layer1(mu1_prev, Sigma1_prev, u1, z1, Q1, R1, rigid_len, r):
    """
    Perform one EKF update for layer 1.
    
    Parameters:
      mu1_prev   : numpy array (6,), previous state of layer 1
      Sigma1_prev: numpy array (6x6), previous covariance for layer 1
      u1         : numpy array, control input for layer 1 [l1, l2, l3]
      z1         : numpy array (3,), measurement from layer 1 IMU (roll, pitch, yaw)
      Q1         : numpy array (6x6), process noise covariance for layer 1
      R1         : numpy array (3x3), measurement noise covariance for layer 1
      rigid_len  : float, rigid extension length for layer 1
      r          : float, PCC model radius parameter
      
    Returns:
      mu1_new    : numpy array (6,), updated state for layer 1
      Sigma1_new : numpy array (6x6), updated covariance for layer 1
    """
    # Prediction: state solely determined by u1 via PCC forward kinematics
    mu1_pred = f_layer1(u1, rigid_len, r)
    # For simplicity, we assume the state prediction does not depend on the previous state:
    F = np.zeros((6,6))
    # Assume orientation part has identity (i.e. direct inheritance in prediction)
    F[3:6, 3:6] = np.eye(3)
    Sigma1_pred = F @ Sigma1_prev @ F.T + Q1  # Essentially Sigma1_pred = Q1
    
    # Observation: we assume the sensor measures the orientation of layer 1.
    z1_pred = mu1_pred[3:6]
    innovation = z1 - z1_pred
    # Observation Jacobian: only orientation (roll, pitch, yaw) is measured.
    H = np.zeros((3,6))
    H[:, 3:6] = np.eye(3)
    S = H @ Sigma1_pred @ H.T + R1
    K = Sigma1_pred @ H.T @ np.linalg.inv(S)
    
    mu1_new = mu1_pred + K @ innovation
    Sigma1_new = (np.eye(6) - K @ H) @ Sigma1_pred
    Sigma1_new = 0.5*(Sigma1_new + Sigma1_new.T)
    
    return mu1_new, Sigma1_new

# ---------------- EKF Update for Layer 2 ----------------

def f_layer2(mu1, u2, rigid_len, r):
    """
    Compute the predicted state for layer 2.
    
    The prediction is computed relative to the corrected state of layer 1.
    
    Parameters:
      mu1       : numpy array (6,), updated state of layer 1
      u2        : numpy array, control input for layer 2 [l1, l2, l3]
      rigid_len : float, rigid extension length for layer 2
      r         : float, PCC model radius parameter
      
    Returns:
      state2_pred : numpy array (6,) = [x2, y2, z2, roll2, pitch2, yaw2]
    """
    # Convert updated layer 1 state to a transformation matrix
    T1 = state_to_transform(mu1)
    T2 = pcc_transform(u2, rigid_len, r)
    T_total = T1 @ T2
    pos2 = T_total[0:3, 3]
    euler2 = rotmat_to_euler(T_total[0:3, 0:3])
    return np.hstack((pos2, euler2))

def ekf_update_layer2(mu2_prev, Sigma2_prev, mu1_updated, u2, z2, Q2, R2, rigid_len, r):
    """
    Perform one EKF update for layer 2.
    
    The prediction for layer 2 is computed based on the updated state of layer 1.
    
    Parameters:
      mu2_prev   : numpy array (6,), previous state of layer 2
      Sigma2_prev: numpy array (6x6), previous covariance for layer 2
      mu1_updated: numpy array (6,), updated state for layer 1 (used as the base)
      u2         : numpy array, control input for layer 2 [l1, l2, l3]
      z2         : numpy array (3,), measurement from layer 2 IMU (roll, pitch, yaw)
      Q2         : numpy array (6x6), process noise covariance for layer 2
      R2         : numpy array (3x3), measurement noise covariance for layer 2
      rigid_len  : float, rigid extension length for layer 2
      r          : float, PCC model radius parameter
      
    Returns:
      mu2_new    : numpy array (6,), updated state for layer 2
      Sigma2_new : numpy array (6x6), updated covariance for layer 2
    """
    mu2_pred = f_layer2(mu1_updated, u2, rigid_len, r)
    F = np.zeros((6,6))
    F[3:6, 3:6] = np.eye(3)
    Sigma2_pred = F @ Sigma2_prev @ F.T + Q2  # Essentially, Sigma2_pred = Q2
    # Observation: measurement from layer 2 IMU.
    z2_pred = mu2_pred[3:6]
    innovation = z2 - z2_pred
    H = np.zeros((3,6))
    H[:, 3:6] = np.eye(3)
    S = H @ Sigma2_pred @ H.T + R2
    K = Sigma2_pred @ H.T @ np.linalg.inv(S)
    
    mu2_new = mu2_pred + K @ innovation
    Sigma2_new = (np.eye(6) - K @ H) @ Sigma2_pred
    Sigma2_new = 0.5*(Sigma2_new + Sigma2_new.T)
    
    return mu2_new, Sigma2_new

# ---------------- Combined Visualization ----------------

def plot_two_layer_pcc(u1, u2, rigid_len, r, mu1, mu2):
    """
    Visualize the two-layer continuum robot.
    
    The function plots:
      - The flexible (arc) portion and rigid extension for layer 1.
      - The arc for layer 2 (transformed using the updated layer 1 state).
      - The base and final end-effector position.
    
    Parameters:
      u1       : numpy array, control input for layer 1 [l1, l2, l3]
      u2       : numpy array, control input for layer 2 [l1, l2, l3]
      rigid_len: float, rigid extension length for each layer
      r        : float, PCC model radius parameter
      mu1      : numpy array (6,), updated state for layer 1
      mu2      : numpy array (6,), updated state for layer 2
    """
    # Compute arc for layer 1 (in its own frame)
    arc1 = arc_curve(u1, r, num_points=100)
    T1 = pcc_transform(u1, rigid_len, r)
    # Rigid extension for layer 1
    phi1 = get_phi(u1[0], u1[1], u1[2])
    beta1 = get_beta(u1[0], u1[1], u1[2], r)
    R1 = compute_rotation_matrix(phi1, beta1)
    t_vals = np.linspace(0, 1, 20)
    rigid1_pts = np.array([arc1[-1] + t * (R1 @ np.array([0,0,1])) * rigid_len for t in t_vals])
    
    # For layer 2, compute arc in its own frame and transform to global using updated mu1
    arc2 = arc_curve(u2, r, num_points=100)
    T1_updated = state_to_transform(mu1)
    arc2_global = np.array([(T1_updated @ np.hstack((pt, 1)))[0:3] for pt in arc2])
    # Rigid extension for layer 2
    phi2 = get_phi(u2[0], u2[1], u2[2])
    beta2 = get_beta(u2[0], u2[1], u2[2], r)
    R2 = compute_rotation_matrix(phi2, beta2)
    rigid2_local = np.array([arc2[-1] + t * (R2 @ np.array([0,0,1])) * rigid_len for t in t_vals])
    rigid2_global = np.array([(T1_updated @ np.hstack((pt, 1)))[0:3] for pt in rigid2_local])
    
    # Final end-effector position is from layer 2 updated state.
    T_total = T1_updated @ pcc_transform(u2, rigid_len, r)
    end_effector = T_total[0:3, 3]
    
    # Plot in 3D.
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(0, 0, 0, color='green', s=50, label="Base")
    ax.plot(arc1[:,0], arc1[:,1], arc1[:,2], 'b-', lw=2, label="Layer 1 Arc")
    ax.plot(rigid1_pts[:,0], rigid1_pts[:,1], rigid1_pts[:,2], 'c--', lw=2, label="Layer 1 Rigid")
    ax.plot(arc2_global[:,0], arc2_global[:,1], arc2_global[:,2], 'm-', lw=2, label="Layer 2 Arc")
    ax.plot(rigid2_global[:,0], rigid2_global[:,1], rigid2_global[:,2], 'y--', lw=2, label="Layer 2 Rigid")
    ax.scatter(end_effector[0], end_effector[1], end_effector[2], color='red', s=50, label="End-Effector")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Two-Layer Stacked PCC Continuum Robot Trajectory (Sequential EKF)")
    ax.legend()
    plt.show()

# ---------------- Example Usage ----------------

if __name__ == "__main__":
    # Initialize previous states and covariances for each layer.
    # Here we assume initial state of both layers is zeros.
    mu1_prev = np.zeros(6)     # Layer 1: [x, y, z, roll, pitch, yaw]
    Sigma1_prev = np.eye(6) * 0.01
    mu2_prev = np.zeros(6)     # Layer 2
    Sigma2_prev = np.eye(6) * 0.01
    
    # Define process noise and measurement noise for each layer.
    Q1 = np.eye(6) * 0.001
    R1 = np.eye(3) * 0.01
    Q2 = np.eye(6) * 0.001
    R2 = np.eye(3) * 0.01
    
    # Example control inputs for each layer.
    u1 = np.array([60.0, 80.0, 100.0])   # Control input for layer 1
    u2 = np.array([70.0, 90.0, 110.0])    # Control input for layer 2
    
    # Example measurements (from IMUs) for each layer (in radians).
    z1 = np.array([0.1, 0.05, 0.0])       # Measurement from layer 1 IMU
    z2 = np.array([0.12, 0.04, -0.02])     # Measurement from layer 2 IMU
    
    # Rigid extension length and PCC model radius parameter.
    rigid_len = 0
    r = 19
    
    # --- EKF Update for Layer 1 ---
    mu1_updated, Sigma1_updated = ekf_update_layer1(mu1_prev, Sigma1_prev, u1, z1, Q1, R1, rigid_len, r)
    
    # --- EKF Update for Layer 2 (using updated layer 1 state as base) ---
    mu2_updated, Sigma2_updated = ekf_update_layer2(mu2_prev, Sigma2_prev, mu1_updated, u2, z2, Q2, R2, rigid_len, r)
    
    # Combine the updated states into an overall state.
    # Overall state: [x, y, z, roll1, pitch1, yaw1, roll2, pitch2, yaw2]
    mu_combined = np.hstack((mu1_updated, mu2_updated))
    
    print("Updated state for Layer 1 (mu1):")
    print(mu1_updated)
    print("Updated state for Layer 2 (mu2):")
    print(mu2_updated)
    print("Combined state (mu):")
    print(mu_combined)
    
    # Visualize the two-layer robot using the updated state of layer 1 as base.
    plot_two_layer_pcc(u1, u2, rigid_len, r, mu1_updated, mu2_updated)
