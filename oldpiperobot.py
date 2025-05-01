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
    lc = np.mean(l)
    rho = beta / lc if lc != 0 else 1e-6
    displacement = calc_displacement(rho, beta, phi)
    rotation = compute_rotation_matrix(phi, beta)
    direction = rotation @ np.array([0, 0, 1])
    return displacement + direction * rigid_len

# ---------------- EKF Functions ----------------

def f(u, rigid_len, r):
    """
    EKF Prediction Function: returns [x, y, z, roll, pitch, yaw].
    """
    pos = length_to_position_single(u, rigid_len, r)
    phi = get_phi(u[0], u[1], u[2])
    beta = get_beta(u[0], u[1], u[2], r)
    R = compute_rotation_matrix(phi, beta)
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    roll  = np.arctan2(R[2,1], R[2,2])
    yaw   = np.arctan2(R[1,0], R[0,0])
    return np.hstack((pos, np.array([roll, pitch, yaw])))


def ekf_update(mu_prev, Sigma_prev, u, z, Q, R_cov, rigid_len, r):
    """
    Perform one EKF iteration.
    """
    mu_pred = f(u, rigid_len, r)
    F = np.eye(6)
    F[3:6, 3:6] = np.eye(3)
    Sigma_pred = F @ Sigma_prev @ F.T + Q
    H = np.zeros((3,6)); H[:,3:6] = np.eye(3)
    S = H @ Sigma_pred @ H.T + R_cov
    K = Sigma_pred @ H.T @ np.linalg.inv(S)
    innovation = z - mu_pred[3:6]
    mu_new = mu_pred + K @ innovation
    Sigma_new = (np.eye(6) - K @ H) @ Sigma_pred
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
    return mu_new, Sigma_new

# ---------------- Constrained EKF Trajectory ----------------

def compute_constrained_trajectory_ekf(u_sequence, rigid_len, r,
                                       initial_base, initial_mode,
                                       pipe_radius, robot_radius,
                                       Q, R_cov, num_samples=100):
    """
    Simulate climbing within a pipe using EKF. Stops early on wall contact
    (considering robot_radius), switches mode, and continues. Returns:
      smooth trajectory (Nx3) and list of (position, mode) change points.
    """
    smooth = []
    mode_points = []
    base = initial_base.copy()
    mode = initial_mode
    mu_rel = np.zeros(6)
    Sigma = np.eye(6) * 0.01

    for u in u_sequence:
        contact = False
        # sample along the motion until either full step or collision
        for a in np.linspace(0, 1, num_samples):
            u_part = u * a
            # EKF update using orientation measurement
            z = f(u_part, rigid_len, r)[3:6] + \
                np.random.multivariate_normal(np.zeros(3), R_cov)
            mu_rel, Sigma = ekf_update(mu_rel, Sigma, u_part, z,
                                       Q, R_cov, rigid_len, r)
            delta = mu_rel[:3]
            pos = base + delta if mode == 'lower_fixed' else base - delta
            # collision check: robot surface touches pipe wall
            if np.hypot(pos[0], pos[1]) + robot_radius >= pipe_radius:
                contact = True
                base = pos.copy()
                mode = 'upper_fixed' if mode == 'lower_fixed' else 'lower_fixed'
                mode_points.append((base.copy(), mode))
                break
            smooth.append(pos)
        if not contact:
            # complete full motion if no collision
            z_full = f(u, rigid_len, r)[3:6] + \
                     np.random.multivariate_normal(np.zeros(3), R_cov)
            mu_rel, Sigma = ekf_update(mu_rel, Sigma, u, z_full,
                                       Q, R_cov, rigid_len, r)
            delta = mu_rel[:3]
            base = base + delta if mode == 'lower_fixed' else base - delta
            mode = 'upper_fixed' if mode == 'lower_fixed' else 'lower_fixed'
            mode_points.append((base.copy(), mode))
        # reset relative state for next segment
        mu_rel = np.zeros(6)
        Sigma = np.eye(6) * 0.01

    return np.array(smooth), mode_points


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

# ---------------- Main Simulation ----------------

if __name__ == '__main__':
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
    rigid_len = 0.0
    r = 19.0
    pipe_radius = 20
    robot_radius = 10
    initial_base = np.array([0.,0.,0.])
    initial_mode = 'lower_fixed'

    traj, modes = compute_constrained_trajectory_ekf(
        u_sequence, rigid_len, r,
        initial_base, initial_mode,
        pipe_radius, robot_radius, Q, R_cov,
        num_samples=200
    )

    # Plot 3D trajectory within pipe
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    # Plot pipe cylinder
    z_min, z_max = traj[:,2].min(), traj[:,2].max()
    Z = np.linspace(z_min, z_max, 50)
    Theta = np.linspace(0, 2*np.pi, 100)
    T, Zm = np.meshgrid(Theta, Z)
    Xp = pipe_radius * np.cos(T)
    Yp = pipe_radius * np.sin(T)
    ax.plot_surface(Xp, Yp, Zm, alpha=0.2)
    
    
    # Compute trajectory
    smooth_traj = compute_smooth_trajectory(
        u_sequence, rigid_len, r,
        initial_base, initial_mode,
        arc_points=200
    )

    # Extract coordinates
    X = smooth_traj[:,0]
    Y = smooth_traj[:,1]
    Z = smooth_traj[:,2]

    # Create tube around path with radius 0.1 m
    
    n_theta = 20
    theta = np.linspace(0, 2*np.pi, n_theta)

    # Initialize mesh grid
    XX = np.zeros((len(X), n_theta))
    YY = np.zeros_like(XX)
    ZZ = np.zeros_like(XX)

    # Generate circular cross-sections in horizontal plane
    for i in range(len(X)):
        XX[i,:] = X[i] + robot_radius * np.cos(theta)
        YY[i,:] = Y[i] + robot_radius * np.sin(theta)
        ZZ[i,:] = Z[i]

    # Plot tube surface
    ax.plot_surface(XX, YY, ZZ, color='cyan', alpha=0.7, linewidth=0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Smooth Trajectory Tube (Radius = 0.1 m) without constraint')
    plt.show()

    # # Plot robot path
    # ax.plot(traj[:,0], traj[:,1], traj[:,2], '-', label='Constrained Path', linewidth = 10)
    # for pos, mode in modes:
    #     ax.scatter(*pos, c='k', s=40)
    #     ax.text(*pos, mode, color='red')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Soft Robot Climbing in 10cm Pipe with EKF')
    # ax.legend()
    # plt.show()
