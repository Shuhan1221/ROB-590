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
    mu_pcc = f(mu_prev, u, rigid_len, r)
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
    ax.plot(full_pts[:,0], full_pts[:,1], full_pts[:,2], 'b-', lw=2, label="Continuum Trajectory")
    
    # Mark the base, the flexible arc end, and the final end-effector position
    ax.scatter(0, 0, 0, color='green', s=50, label="Base")
    ax.scatter(flex_end[0], flex_end[1], flex_end[2], color='orange', s=50, label="Arc End")
    ax.scatter(end_pos[0], end_pos[1], end_pos[2], color='red', s=50, label="End-Effector")
    
    # # Plot the base (end-effector position from PCC prediction)
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