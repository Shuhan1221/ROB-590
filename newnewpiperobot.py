import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------- PCC Forward Kinematics Functions ----------------

def compute_rotation_matrix(phi, beta):
    """
    Compute the rotation matrix given a directional angle (phi) and a bending angle (beta).
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
    """
    return (1 / rho) * np.array([(1 - np.cos(beta)) * np.cos(phi),
                                 (1 - np.cos(beta)) * np.sin(phi),
                                 np.sin(beta)])


def get_beta(l1, l2, l3, r):
    """
    Compute the bending angle beta from segment lengths.
    """
    return 2 * np.sqrt(l1**2 + l2**2 + l3**2 - l1*l2 - l1*l3 - l2*l3) / (3 * r)


def get_phi(l1, l2, l3):
    """
    Compute the directional angle phi from segment lengths.
    """
    return np.arctan2(3 * (l2 - l3), np.sqrt(3) * (l2 + l3 - 2 * l1))


def arc_curve(l, r, num_points=100):
    """
    Compute points along the flexible arc portion of the continuum robot.
    """
    phi = get_phi(l[0], l[1], l[2])
    beta = get_beta(l[0], l[1], l[2], r)
    lc = np.mean(l)
    rho = beta / lc if lc != 0 else 1e-6

    t_vals = np.linspace(0, 1, num_points)
    pts = []
    for t in t_vals:
        disp_t = (1 / rho) * np.array([
            (1 - np.cos(t * beta)) * np.cos(phi),
            (1 - np.cos(t * beta)) * np.sin(phi),
            np.sin(t * beta)
        ])
        pts.append(disp_t)
    return np.array(pts)

def length_to_position_single(l, rigid_len, r):
    """
    Compute the end-effector position for a single segment based on control input.
    """
    phi = get_phi(l[0], l[1], l[2])
    beta = get_beta(l[0], l[1], l[2], r)
    lc = np.mean(l)  # approximate flexible length as the average
    rho = beta / lc if lc != 0 else 1e-6
    displacement = calc_displacement(rho, beta, phi)
    rotation = compute_rotation_matrix(phi, beta)
    direction = rotation @ np.array([0, 0, 1])  # local z-axis transformed
    end_position = displacement + direction * rigid_len
    return end_position


def rotmat_to_euler(l, r):
    """
    Convert the rotation matrix (computed from control input l) to Euler angles (roll, pitch, yaw)
    using the ZYX convention.
    """
    phi = get_phi(l[0], l[1], l[2])
    beta = get_beta(l[0], l[1], l[2], r)
    a = np.cos(beta) * np.cos(phi)**2 + np.sin(phi)**2
    b = (-1 + np.cos(beta)) * np.cos(phi) * np.sin(phi)
    c = np.cos(phi) * np.sin(beta)
    d = np.sin(beta) * np.sin(phi)
    e = np.cos(beta)
    f_val = np.cos(beta) * np.sin(phi)**2 + np.cos(phi)**2
    R_mat = np.array([[a, b, c],
                      [b, f_val, d],
                      [-c, -d, e]])
    pitch = np.arctan2(-R_mat[2,0], np.sqrt(R_mat[0,0]**2 + R_mat[1,0]**2))
    roll = np.arctan2(R_mat[2,1], R_mat[2,2])
    yaw = np.arctan2(R_mat[1,0], R_mat[0,0])
    return np.array([roll, pitch, yaw])


def f(u, rigid_len, r):
    """
    EKF Prediction Function: Compute the predicted state from control input.
    The state vector is [x, y, z, roll, pitch, yaw].
    """
    pos = length_to_position_single(u, rigid_len, r)
    orientation = rotmat_to_euler(u, r)
    return np.hstack((pos, orientation))


def h(mu):
    """
    EKF Observation Function: Returns the orientation part of the state.
    """
    return mu[3:6]


def ekf_update(mu_prev, Sigma_prev, u, z, Q, R_cov, rigid_len, r):
    """
    Perform one iteration of the EKF update.
    """
    mu_pred = f(u, rigid_len, r)
    F = np.zeros((6, 6))
    F[3:6, 3:6] = np.eye(3)
    Sigma_pred = F @ Sigma_prev @ F.T + Q
    z_pred = h(mu_pred)
    innovation = z - z_pred
    H = np.zeros((3, 6))
    H[:, 3:6] = np.eye(3)
    S = H @ Sigma_pred @ H.T + R_cov
    K = Sigma_pred @ H.T @ np.linalg.inv(S)
    mu_new = mu_pred + K @ innovation
    Sigma_new = (np.eye(6) - K @ H) @ Sigma_pred
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
    return mu_new, Sigma_new


def simulate_climbing_no_projection(u_sequence, rigid_len, r, Q, R_cov,
                                    initial_base=np.zeros(3), initial_mode="lower_fixed"):
    """
    Simulate the climbing process of the soft robot without projecting the positions.
    """
    N = len(u_sequence)
    state_history = []
    base_history = []
    mode_history = []

    mu_rel = np.zeros(6)
    Sigma = np.eye(6) * 0.01
    base = initial_base.copy()
    mode = initial_mode

    # initial base
    base_history.append(base.copy())
    mode_history.append(mode)

    for u in u_sequence:
        # EKF update
        true_orient = f(u, rigid_len, r)[3:6]
        z = true_orient + np.random.multivariate_normal(np.zeros(3), R_cov)
        mu_rel, Sigma = ekf_update(mu_rel, Sigma, u, z, Q, R_cov, rigid_len, r)

        # compute absolute pos
        if mode == "lower_fixed":
            mu_abs = base + mu_rel[0:3]
            base = mu_abs.copy()
            mode = "upper_fixed"
        else:
            mu_abs = base - mu_rel[0:3]
            base = mu_abs.copy()
            mode = "lower_fixed"

        state_history.append(mu_abs.copy())
        base_history.append(base.copy())
        mode_history.append(mode)
        mu_rel = np.zeros(6)

    return np.array(state_history), np.array(base_history), mode_history


def compute_smooth_trajectory(u_sequence, rigid_len, r, initial_base, initial_mode,
                              num_samples_per_step=10, arc_points=50):
    """
    Generate a smooth, physically accurate trajectory by sampling arc segments.
    """
    smooth = []
    base = initial_base.copy()
    mode = initial_mode

    for u in u_sequence:
        # sample along arc for each control input
        arc_pts = arc_curve(u, r, num_points=arc_points)
        for pt in arc_pts:
            pos = base + pt if mode == "lower_fixed" else base - pt
            smooth.append(pos)
        # advance base by full arc end plus rigid length
        full_disp = arc_pts[-1] + np.array([0, 0, rigid_len])
        base = base + full_disp if mode == "lower_fixed" else base - full_disp
        mode = "upper_fixed" if mode == "lower_fixed" else "lower_fixed"

    return np.array(smooth)

# ---------------- Main Simulation and Plotting ----------------

if __name__ == "__main__":
    # Parameters
    Q = np.eye(6) * 0.001
    R_cov = np.eye(3) * 0.01
    # u_sequence = [
    #     np.array([20.0, 40.0, 60.0]),
    #     np.array([80.0, 10.0, 90.0]),
    #     np.array([45.0, 75.0, 120.0]),
    #     np.array([100.0, 20.0, 50.0]),
    #     np.array([60.0, 110.0, 30.0])
    # ]
    u_sequence = [
        np.array([40., 30., 40.]),
        np.array([20., 20., 30.]),
        np.array([50., 40.,50.]),
        np.array([10.,20., 10.]),
        np.array([60.,70., 60.])
    ]
    rigid_len = 0
    r = 19
    initial_base = np.array([0, 0, 0])
    initial_mode = "lower_fixed"

    # Run simulation
    state_history, base_history, mode_history = simulate_climbing_no_projection(
        u_sequence, rigid_len, r, Q, R_cov, initial_base, initial_mode
    )

    # Generate smooth trajectory
    smooth_traj = compute_smooth_trajectory(
        u_sequence, rigid_len, r, initial_base, initial_mode,
        num_samples_per_step=100
    )

    # Plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(smooth_traj[:, 0], smooth_traj[:, 1], smooth_traj[:, 2], '-', label="Smooth Curved Trajectory")
    ax.scatter(base_history[:, 0], base_history[:, 1], base_history[:, 2], color='k', s=40, label="Cycle Base Points")
    ax.scatter(base_history[-1, 0], base_history[-1, 1], base_history[-1, 2], color='g', s=40, label="Final Points")
    # Annotate each cycle
    for i, pos in enumerate(base_history[:-1]):
        ax.text(pos[0], pos[1], pos[2], f"Cycle {i}\n({mode_history[i]})", fontsize=8, color='red')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Soft Robot Climbing without Constraint")
    ax.legend()
    plt.show()
