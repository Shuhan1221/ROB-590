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

def length_to_position_single(l, rigid_len, r):
    """
    Compute the end-effector position for a single segment.
    
    Parameters:
      l         : list - Three segment lengths [l1, l2, l3]
      rigid_len : float - Length of the rigid body extension
      r         : float - Radius parameter
      
    Returns:
      3D end-effector position (numpy array)
    """
    # Compute the directional and bending angles
    phi = get_phi(l[0], l[1], l[2])
    beta = get_beta(l[0], l[1], l[2], r)
    lc = np.mean(l)  # average segment length
    rho = beta / lc  # curvature

    # Compute displacement and rotation of the flexible (arc) portion
    displacement = calc_displacement(rho, beta, phi)
    rotation = compute_rotation_matrix(phi, beta)
    direction = rotation @ np.array([0, 0, 1])  # transformed z-axis
    
    # Final position = flexible arc end + rigid extension along computed direction
    end_position = displacement + direction * rigid_len
    return end_position
  

def rotmat_to_euler(l,r):
    """
    Convert a rotation matrix R (3x3) to Euler angles (roll, pitch, yaw)
    using the ZYX convention.
    
    Parameters:
      R : 3x3 numpy array
      
    Returns:
      euler : numpy array [roll, pitch, yaw] (in radians)
    """
    
    phi = get_phi(l[0], l[1], l[2])
    beta = get_beta(l[0], l[1], l[2], r)
    
    a = np.cos(beta) * np.cos(phi)**2 + np.sin(phi)**2
    b = (-1 + np.cos(beta)) * np.cos(phi) * np.sin(phi)
    c = np.cos(phi) * np.sin(beta)
    d = np.sin(beta) * np.sin(phi)
    e = np.cos(beta)
    f = np.cos(beta) * np.sin(phi)**2 + np.cos(phi)**2
    
    R = np.array([[a, b, c], [b, f, d], [-c, -d, e]])
    
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    # cos_pitch = np.cos(pitch)
    roll = np.arctan2(R[2,1],R[2,2])
    yaw = np.arctan2(R[1,0],R[0,0])
    
    # if abs(R[2,0]) < 1:
    #     pitch = -np.arctan2(R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    #     # cos_pitch = np.cos(pitch)
    #     roll = np.arctan2(R[2,1],R[2,2])
    #     yaw = np.arctan2(R[1,0],R[0,0])
    # else:
    #     yaw = 0
    #     if R[2,0] <= -1:
    #         pitch = np.pi/2
    #         roll = np.arctan2(R[0,1], R[0,2])
    #     else:
    #         pitch = -np.pi/2
    #         roll = np.arctan2(-R[0,1], -R[0,2])
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
    return Rx @ Ry @ Rz
  
def arc_curve(l, r, num_points=100):
    """
    Compute the set of points along the flexible (arc) portion of the continuum robot.
    
    Parameters:
      l         : list - Three segment lengths [l1, l2, l3]
      r         : float - Radius parameter
      num_points: int   - Number of points along the arc
    
    Returns:
      A numpy array of shape (num_points, 3) representing points along the arc.
    """
    phi = get_phi(l[0], l[1], l[2])
    beta = get_beta(l[0], l[1], l[2], r)
    lc = np.mean(l)
    rho = beta / lc

    t_vals = np.linspace(0, 1, num_points)
    points = []
    for t in t_vals:
        # Parameterize the arc from t=0 to t=1
        disp_t = (1 / rho) * np.array([(1 - np.cos(t * beta)) * np.sin(phi),
                                       (1 - np.cos(t * beta)) * np.cos(phi),
                                       np.sin(t * beta)])
        points.append(disp_t)
    return np.array(points)

# ---------------- EKF Functions ----------------

def f(u, rigid_len, r):
    """
    EKF Prediction function: mu_pred = f(mu_prev, u)
    
    The function computes:
      - [x, y, z] via the PCC forward kinematics from control u = [l1, l2, l3]
      - [roll, pitch, yaw] are inherited from the previous state.
    
    Parameters:
      mu_prev   : numpy array, previous state [x, y, z, roll, pitch, yaw]
      u         : numpy array, control input [l1, l2, l3]
      rigid_len : float, length of the rigid extension
      r         : float, PCC model's radius parameter
      
    Returns:
      mu_pred   : numpy array, predicted state vector
    """
    pos = length_to_position_single(u, rigid_len, r)
    direction = rotmat_to_euler(u,r)
    return np.hstack((pos,direction ))

def h(mu):
    """
    EKF Observation function: z_pred = h(mu)
    
    Here we assume that the sensor directly measures the orientation,
    i.e., [roll, pitch, yaw] from the state.
    
    Parameters:
      mu : numpy array, state vector
      
    Returns:
      Measurement vector corresponding to [roll, pitch, yaw]
    """
    return mu[3:6]

def ekf_update(mu_prev, Sigma_prev, u, z, Q, R, rigid_len, r):
    """
    Perform one iteration of the EKF update.
    
    Parameters:
      mu_prev   : numpy array, previous state mean [x, y, z, roll, pitch, yaw]
      Sigma_prev: numpy array (6x6), previous state covariance
      u         : numpy array, control input [l1, l2, l3]
      z         : numpy array, measurement [roll_meas, pitch_meas, yaw_meas]
      Q         : numpy array (6x6), process noise covariance
      R         : numpy array (3x3), measurement noise covariance
      rigid_len : float, length of the rigid extension for PCC kinematics
      r         : float, radius parameter for the PCC model
      
    Returns:
      mu_new    : numpy array, updated state mean
      Sigma_new : numpy array, updated state covariance matrix
    """
    # ----- Prediction Step -----
    mu_pred = f(u, rigid_len, r)
    
    # Jacobian of f with respect to mu_prev.
    # Since [x,y,z] are completely determined by u (not mu_prev), we set that block to zeros.
    # The orientation part is directly inherited, hence its derivative is an identity.
    F = np.zeros((6, 6))
    F[3:6, 3:6] = np.eye(3)
    
    Sigma_pred = F @ Sigma_prev @ F.T + Q
    
    # ----- Update Step -----
    z_pred = h(mu_pred)
    innovation = z - z_pred
    
    # Jacobian of the observation function h with respect to mu.
    # Only orientation is measured, so H has zeros for [x,y,z] and identity for [roll, pitch, yaw].
    H = np.zeros((3, 6))
    H[:, 3:6] = np.eye(3)
    
    S = H @ Sigma_pred @ H.T + R
    K = Sigma_pred @ H.T @ np.linalg.inv(S)
    
    mu_new = mu_pred + K @ innovation
    Sigma_new = (np.eye(6) - K @ H) @ Sigma_pred
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
    
    return mu_new, Sigma_new

# ---------------- Visualization Functions ----------------

def plot_arc_orientation(u, rigid_len, r, z, mu_prev, Sigma_prev, Q, R, arrow_length=1):
    # Compute the flexible arc points
    arc_pts = arc_curve(u, r, num_points=100)
    
    # Get the full end-effector position (including rigid extension)
    end_pos = length_to_position_single(u, rigid_len, r)
    
    # Compute the orientation direction from the PCC model (same as in length_to_position_single)
    phi = get_phi(u[0], u[1], u[2])
    beta = get_beta(u[0], u[1], u[2], r)
    rotation = compute_rotation_matrix(phi, beta)
    direction = rotation @ np.array([0, 0, 1])
    
    # The flexible arc end is the last point in arc_pts
    flex_end = arc_pts[-1, :]
    
    # Generate points for the rigid extension as a straight line
    t_vals = np.linspace(0, 1, 20)
    rigid_pts = np.array([flex_end + t * direction * rigid_len for t in t_vals])
    
    # Combine all points for full plot (flexible arc followed by rigid extension)
    full_pts = np.vstack((arc_pts, rigid_pts))
    
        # Compute PCC predicted state using forward kinematics (f_layer1 must be defined elsewhere)
    mu_pcc = f(u, rigid_len, r)
    pcc_orient = mu_pcc[3:6]
    
    # Perform EKF update (ekf_update_layer1 must be defined elsewhere)
    mu_updated, Sigma_updated = ekf_update(mu_prev, Sigma_prev, u, z, Q, R, rigid_len, r)
    ekf_orient = mu_updated[3:6]
    
    # Sensor measurement (raw IMU input)
    sensor_orient = z
    
    # Convert each orientation (Euler angles) to a rotation matrix
    R_pcc = euler_to_rotmat(*pcc_orient)
    R_sensor = euler_to_rotmat(*sensor_orient)
    R_ekf = euler_to_rotmat(*ekf_orient)
    
    # Use the third column (local z-axis) as the arrow direction for each case.
    arrow_pcc = R_pcc[:, 2]
    arrow_sensor = R_sensor[:, 2]
    arrow_ekf = R_ekf[:, 2]
    
    
    # Use the end-effector position (from PCC prediction) as the base point for the arrows.
    base_position = mu_pcc[0:3]
    
    # # Plot in 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the full trajectory of the continuum segment
    ax.plot(full_pts[:,0], full_pts[:,1], full_pts[:,2], 'y-', lw=2, label="Continuum Trajectory")
    
    # Mark the base, the flexible arc end, and the final end-effector position
    ax.scatter(0, 0, 0, color='black', s=50, label="Base")
    # ax.scatter(flex_end[0], flex_end[1], flex_end[2], color='orange', s=50, label="Arc End")
    ax.scatter(end_pos[0], end_pos[1], end_pos[2], color='orange', s=50, label="End-Effector")
    
    # Plot the base (end-effector position from PCC prediction)
    # ax.scatter(base_position[0], base_position[1], base_position[2],
    #            color='black', s=50, label="PCC End-Effector Position")
    
    # Plot arrows using quiver (x, y, z, u, v, w)
    ax.quiver(base_position[0], base_position[1], base_position[2],
              arrow_pcc[0], arrow_pcc[1], arrow_pcc[2],
              length=arrow_length, color='blue', label='PCC Orientation')
    
    ax.quiver(base_position[0], base_position[1], base_position[2],
              arrow_sensor[0], arrow_sensor[1], arrow_sensor[2],
              length=arrow_length, color='green', label='IMU Measurement')
    
    ax.quiver(base_position[0], base_position[1], base_position[2],
              arrow_ekf[0], arrow_ekf[1], arrow_ekf[2],
              length=arrow_length, color='red', label='EKF Corrected')
    
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("PCC Continuum Robot Trajectory")
    ax.legend()
    plt.show()

# ---------------- Example Usage ----------------

if __name__ == "__main__":
    # Initialize EKF state: initial position at origin and zero orientation.
    mu_prev = np.array([0, 0, 0, 0.0, 0.0, 0.0])  # [x, y, z, roll, pitch, yaw]
    Sigma_prev = np.eye(6) * 0.01  # initial covariance
    
    # Define process noise and measurement noise covariances.
    Q = np.eye(6) * 0.001
    R = np.eye(3) * 0.01
    
    # Example control input: actuator lengths.
    l1, l2, l3 = 60.0, 80.0, 100.0
    u_k = np.array([l1, l2, l3])
    
    # PCC forward kinematics parameters: rigid extension length and radius parameter.
    rigid_len = 0  # Length of the rigid extension.
    r = 19         # Radius parameter for the PCC model.
    
    # Example measurement (sensor-measured orientation, in radians)
    z_k = np.array([0.1, 0.05, 0.2])
    
    # Perform one EKF update.
    mu_updated, Sigma_updated = ekf_update(mu_prev, Sigma_prev, u_k, z_k, Q, R, rigid_len, r)
    
    print("Updated state mu:")
    print(mu_updated)
    print("Updated covariance Sigma:")
    print(Sigma_updated)
    
    # Visualize the PCC arc trajectory (flexible portion + rigid extension).   
    # plot_pcc_arc(u_k, rigid_len, r)
    # plot_orientation_comparison(u_k, rigid_len, r, z_k, mu, Sigma, Q, R, arrow_length=1)
    plot_arc_orientation(u_k, rigid_len, r, z_k, mu_prev, Sigma_prev, Q, R, arrow_length=10)