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

# ---------- Modified Plotting Functions that accept axis (ax) ----------

def plot_endpoints_with_coordinate_frames(u1, u2, rigid_len, r,
                                          z1, z2,
                                          mu1_prev, Sigma1_prev, Q1, R1,
                                          mu2_prev, Sigma2_prev, Q2, R2,
                                          arrow_length=5):
    """
    Create a single 3D plot that shows:
      - The base (starting point).
      - The endpoint of layer 1.
      - The overall endpoint (from layer 2).
      - At each endpoint, the coordinate frame axes (computed from the updated orientation).
    
    Parameters:
      u1, u2        : Control inputs for layer 1 and layer 2.
      rigid_len     : Rigid extension length.
      r             : PCC model's radius parameter.
      z1, z2        : Sensor (IMU) measurements for layer 1 and layer 2 (each [roll, pitch, yaw]).
      mu1_prev, Sigma1_prev, Q1, R1 : EKF parameters for layer 1.
      mu2_prev, Sigma2_prev, Q2, R2 : EKF parameters for layer 2.
      arrow_length  : Length of the coordinate frame arrows.
    """
    # Update layer 1 state with EKF
    mu1_updated, _ = ekf_update_layer1(mu1_prev, Sigma1_prev, u1, z1, Q1, R1, rigid_len, r)
    # Update layer 2 state with EKF using updated layer 1 state as base.
    mu2_updated, _ = ekf_update_layer2(mu2_prev, Sigma2_prev, mu1_updated, u2, z2, Q2, R2, rigid_len, r)
    
    # Compute endpoints:
    base = np.array([0, 0, 0])  # Base point at origin.
    endpoint1 = mu1_updated[0:3]  # Layer 1 endpoint.
    endpoint2 = mu2_updated[0:3]  # Overall endpoint (Layer 2).
    
    # Compute rotation matrices (local coordinate frames) from updated orientations.
    R1 = euler_to_rotmat(*mu1_updated[3:6])
    R2 = euler_to_rotmat(*mu2_updated[3:6])
    
    # For each endpoint, we will plot three arrows representing the local x, y, z axes.
    # For layer 1, use colors: red (x), green (y), blue (z).
    # For layer 2, use colors: magenta (x), cyan (y), yellow (z).
    
    # Create the figure and 3D axis.
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot base and endpoints.
    ax.scatter(base[0], base[1], base[2], color='black', s=50, label="Base")
    ax.scatter(endpoint1[0], endpoint1[1], endpoint1[2], color='orange', s=50, label="Layer1 Endpoint")
    ax.scatter(endpoint2[0], endpoint2[1], endpoint2[2], color='red', s=50, label="Layer2 Endpoint")
    
    # Plot coordinate frame for Layer 1 endpoint.
    # Local x, y, z axes are the columns of R1.
    ax.quiver(endpoint1[0], endpoint1[1], endpoint1[2],
              R1[0,0], R1[1,0], R1[2,0], length=arrow_length, color='red', label="Layer1 X-axis")
    ax.quiver(endpoint1[0], endpoint1[1], endpoint1[2],
              R1[0,1], R1[1,1], R1[2,1], length=arrow_length, color='green', label="Layer1 Y-axis")
    ax.quiver(endpoint1[0], endpoint1[1], endpoint1[2],
              R1[0,2], R1[1,2], R1[2,2], length=arrow_length, color='blue', label="Layer1 Z-axis")
    
    # Plot coordinate frame for Layer 2 endpoint.
    ax.quiver(endpoint2[0], endpoint2[1], endpoint2[2],
              R2[0,0], R2[1,0], R2[2,0], length=arrow_length, color='magenta', label="Layer2 X-axis")
    ax.quiver(endpoint2[0], endpoint2[1], endpoint2[2],
              R2[0,1], R2[1,1], R2[2,1], length=arrow_length, color='cyan', label="Layer2 Y-axis")
    ax.quiver(endpoint2[0], endpoint2[1], endpoint2[2],
              R2[0,2], R2[1,2], R2[2,2], length=arrow_length, color='yellow', label="Layer2 Z-axis")
    
    # Set labels and title.
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Base and Two Endpoints with Local Coordinate Frames (Roll, Pitch, Yaw Directions)")
    ax.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.show()

# ---------- Example Usage ----------
if __name__ == "__main__":
    # EKF parameters for layer 1.
    mu1_prev = np.zeros(6)     # [x, y, z, roll, pitch, yaw]
    Sigma1_prev = np.eye(6) * 0.01
    Q1 = np.eye(6) * 0.001
    R1 = np.eye(3) * 0.01
    
    # EKF parameters for layer 2.
    mu2_prev = np.zeros(6)
    Sigma2_prev = np.eye(6) * 0.01
    Q2 = np.eye(6) * 0.001
    R2 = np.eye(3) * 0.01
    
    # Example control inputs for the two layers.
    u1 = np.array([60.0, 80.0, 100.0])   # Layer 1 control input.
    u2 = np.array([70.0, 90.0, 110.0])    # Layer 2 control input.
    
    # Example sensor (IMU) measurements for the two layers (in radians).
    z1 = np.array([0.1, 0.05, 0.0])       # Layer 1 measurement.
    z2 = np.array([0.12, 0.04, -0.02])     # Layer 2 measurement.
    
    # Rigid extension length and PCC model radius parameter.
    rigid_len = 0   # For a pure flexible model.
    r = 19
    
    # Call the master function to create a single plot.
    plot_endpoints_with_coordinate_frames(u1, u2, rigid_len, r,
                                          z1, z2,
                                          mu1_prev, Sigma1_prev, Q1, R1,
                                          mu2_prev, Sigma2_prev, Q2, R2,
                                          arrow_length=5)