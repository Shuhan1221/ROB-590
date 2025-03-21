import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONSTANT CURVATURE (PCC) MODEL
# -----------------------------------------------------------------------------
# Given an end-effector position p = [x, y, z] (in mm), this function computes
# three actuator values (q1, q2, q3) based on the PCC assumption.
# In your system, these q-values (which you refer to as the three “corners”
# of an equilateral triangle) are obtained via encoder measurements.
# =============================================================================
def constant_curvature(p):
    """
    Compute actuator values using the constant curvature (PCC) assumption.

    Args:
        p: 3-element vector [x, y, z] representing the end-effector position (mm)

    Returns:
        numpy array [q1, q2, q3] corresponding to the three actuator measurements.
    """
    x, y, z = p[0], p[1], p[2]
    d = 19    # Distance from the chamber center to base center (mm)
    aa = 0    # Fixed distance from the base to the chamber cross-section

    # Special case: if p is at the origin, return straight lengths
    if x == 0 and y == 0:
        q1 = (z - aa)
        q2 = (z - aa)
        q3 = (z - aa)
        return np.array([q1, q2, q3])

    # Calculate angle in the XY-plane
    phi = np.arctan2(y, x)
    # Initial guess for the nonlinear solver
    initial_guess = [x, y, z, 1.0, 0.1]

    def nonlinear_equations(variables):
        x0, y0, z0, k0, theta0 = variables
        f = np.zeros(5)
        f[0] = 2 * np.sqrt(x0**2 + y0**2) / (x0**2 + y0**2 + z0**2) - k0
        f[1] = np.arccos(np.clip(1 - k0 * np.sqrt(x0**2 + y0**2), -1, 1)) - theta0
        f[2] = x0 + aa * np.sin(theta0) * np.cos(phi) - x
        f[3] = y0 + aa * np.sin(theta0) * np.sin(phi) - y
        f[4] = z0 + aa * np.cos(theta0) - z
        return f

    try:
        solution = fsolve(nonlinear_equations, initial_guess, xtol=1e-12, maxfev=int(1e8))
    except Exception as e:
        print("Fsolve failed with exception:", e)
        return np.array([np.nan, np.nan, np.nan])
    
    x1, y1, z1, k, theta = solution

    # Compute curvature radii for each chamber
    R1 = 1 / k - d * np.sin(phi)
    R2 = 1 / k + d * np.sin(np.pi / 3 + phi)
    R3 = 1 / k - d * np.cos(np.pi / 6 + phi)

    # Calculate actuator lengths (or effective measures)
    q1 = float(theta * R1)
    q2 = float(theta * R2)
    q3 = float(theta * R3)

    return np.array([q1, q2, q3])

# =============================================================================
# 2. COMPUTE ROUGH END-EFFECTOR POSITION FROM ACTUATOR MEASUREMENTS
# -----------------------------------------------------------------------------
# We assume that the three actuators (whose measured values are q1, q2, q3)
# are arranged at 0°, 120°, and 240° in the horizontal plane.
# We compute the (x, y) coordinates of each actuator’s tip by assuming that
# the measured value q acts as a radial distance. Then the rough end-effector
# horizontal position is taken as the centroid (average) of these three points.
#
# The rough z coordinate is assumed to be provided directly (e.g., by the input p).
# =============================================================================
def rough_end_effector_position(q, z_value):
    """
    Compute the rough end-effector position given actuator measurements.

    Args:
        q: 3-element array containing [q1, q2, q3] from constant_curvature(p)
        z_value: The z-coordinate (assumed to be the same for the three)

    Returns:
        3x1 numpy array representing the rough [x, y, z] end-effector position.
    """
    # Actuators are arranged at 0, 120, 240 degrees (in radians)
    angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
    pts = np.zeros((3, 2))
    for i in range(3):
        pts[i, 0] = q[i] * np.cos(angles[i])
        pts[i, 1] = q[i] * np.sin(angles[i])
    centroid_xy = np.mean(pts, axis=0)
    # Combine horizontal centroid with given z_value
    return np.array([[centroid_xy[0]], [centroid_xy[1]], [z_value]])

# =============================================================================
# 3. EKF MEASUREMENT MODELS
#
# Sensor 1 (encoder/PCC) measurement:
#   We assume the rough end-effector position is given by the PCC fusion,
#   so the measurement model is simply h1(x) = x.
#
# Sensor 2 (IMU) measurement:
#   The 6-axis IMU returns 6 measurements. Here, we model the measurement as:
#     - The first three components are an orientation measurement: Rot * x + C2.
#     - The last three components are an acceleration measurement: t * x.
#
# =============================================================================
def measurement_model_1(x, C1):
    """
    Measurement model for sensor 1 (encoder/PCC).
    Assumes h1(x) = I * x + C1.
    
    Args:
        x: 3x1 state vector (end-effector position)
        C1: 3x1 constant offset
        
    Returns:
        3x1 predicted measurement.
    """
    return x + C1

def measurement_model_2(x, C2, Rot, t):
    """
    Measurement model for sensor 2 (6-axis IMU).
    Returns a 6x1 vector:
      - First 3 components: orientation measurement = Rot * x + C2.
      - Last 3 components: acceleration measurement = t * x.
    
    Args:
        x: 3x1 state vector
        C2: 3x1 offset vector
        Rot: 3x3 rotation matrix (IMU mounting)
        t: scalar multiplier for acceleration
        
    Returns:
        6x1 predicted measurement.
    """
    orientation = Rot @ x + C2
    acceleration = t * x
    return np.vstack((orientation, acceleration))

def measurement_Jacobian_1(x):
    """
    Jacobian for sensor 1 measurement model h1(x)=x.
    
    Args:
        x: state vector (unused)
        
    Returns:
        3x3 identity matrix.
    """
    return np.eye(3)

def measurement_Jacobian_2(x, Rot, t):
    """
    Jacobian for sensor 2 measurement model.
    The derivative of (Rot*x + C2) with respect to x is Rot.
    The derivative of (t*x) with respect to x is t*I.
    
    Args:
        x: state vector (unused)
        Rot: 3x3 rotation matrix
        t: scalar
        
    Returns:
        6x3 Jacobian matrix.
    """
    J_orientation = Rot
    J_acceleration = t * np.eye(3)
    return np.vstack((J_orientation, J_acceleration))

# =============================================================================
# 4. EXTENDED KALMAN FILTER (EKF) CLASS
#
# The EKF state is the refined end-effector position x = [x, y, z].
# The filter fuses two measurements:
#    - Sensor 1: rough position (from PCC) via h1(x) = x.
#    - Sensor 2: 6-axis IMU via h2(x) = [Rot*x + C2; t*x].
#
# The batch correction stacks these two measurements.
# =============================================================================
class ExtendedKalmanFilter:
    def __init__(self, data):
        """
        Initialize the EKF with sensor parameters and initial state.
        
        Args:
            data: dictionary containing:
                'z_1'  : samples from sensor 1 (encoder/PCC) measurements
                'z_2'  : samples from sensor 2 (IMU) measurements
                'C_1'  : offset for sensor 1 (3x1)
                'C_2'  : offset for sensor 2 (3x1)
                'Kf_1' : scaling for sensor 1 (set to identity)
                'Kf_2' : scaling for sensor 2 (set to identity)
                'R'    : rotation matrix for sensor 2 (IMU)
                't'    : scalar multiplier for acceleration measurement (IMU)
        """
        # For sensor 1: measurement model is h1(x)=x so Kf_1 = I and C_1 = data['C_1']
        self.F = np.eye(3)  # State transition Jacobian (constant state)
        self.R1 = np.cov(data['z_1'], rowvar=False)  # Covariance for sensor 1 measurements
        self.R2 = np.cov(data['z_2'], rowvar=False)  # Covariance for sensor 2 measurements
        self.Q = np.array([[0.03, 0.02, 0.01],
                           [0.02, 0.04, 0.01],
                           [0.01, 0.01, 0.05]])   # Process noise covariance
        self.W = np.eye(3)  # Process noise Jacobian
        
        self.C_1 = data['C_1']   # Offset for sensor 1 (3x1)
        self.C_2 = data['C_2']   # Offset for sensor 2 (3x1)
        self.Kf_1 = data['Kf_1'] # For sensor 1 (set as identity)
        self.Kf_2 = data['Kf_2'] # For sensor 2 (set as identity)
        self.Rot = data['R']     # Rotation matrix for sensor 2 (IMU)
        self.t = data['t']       # Scalar for acceleration measurement
        
        # Stack noise covariances for batch update
        self.R_stack = block_diag(self.R1, self.R2)
        
        # Measurement models and their Jacobians
        # Sensor 1:
        self.h1 = lambda x: self.Kf_1 @ x + self.C_1  # with Kf_1 = I, C_1 = 0 => h1(x)=x
        self.H1 = measurement_Jacobian_1
        # Sensor 2:
        self.h2 = lambda x: measurement_model_2(x, self.C_2, self.Rot, self.t)
        self.H2 = lambda x: measurement_Jacobian_2(x, self.Rot, self.t)
        
        # Initial state (assume an initial guess for end-effector position in meters)
        self.x = np.array([[0.12], [0.09], [1.5]])
        self.Sigma = np.eye(3)
    
    def prediction(self):
        """
        EKF prediction step.
        """
        self.x = process_model(self.x)
        self.Sigma = self.F @ self.Sigma @ self.F.T + self.W @ self.Q @ self.W.T

    def correction_batch(self, z_stack):
        """
        EKF correction using the stacked measurement from sensor 1 and sensor 2.
        
        Args:
            z_stack: Stacked measurement vector (9x1), with first 3 elements from sensor 1
                     and next 6 from sensor 2.
        """
        # Predicted measurements
        z_hat1 = self.h1(self.x)  # Sensor 1 prediction (3x1)
        z_hat2 = self.h2(self.x)  # Sensor 2 prediction (6x1)
        z_hat_stack = np.vstack((z_hat1, z_hat2))
        
        # Stacked Jacobian
        H1 = self.H1(self.x)       # 3x3
        H2 = self.H2(self.x)       # 6x3
        H = np.vstack((H1, H2))     # 9x3
        
        # Innovation
        v = z_stack - z_hat_stack
        S = H @ self.Sigma @ H.T + self.R_stack
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ v
        I = np.eye(self.x.shape[0])
        temp = I - K @ H
        self.Sigma = temp @ self.Sigma @ temp.T + K @ self.R_stack @ K.T

# =============================================================================
# 5. MAIN SIMULATION
#
# In this simulation we assume a “true” end-effector position p_true (in mm)
# that changes over time. At each time step:
#   (a) The PCC model (constant_curvature) is used to compute actuator values q.
#   (b) The rough end-effector position is computed as the centroid of the 
#       three actuator points (with the given z value).
#   (c) A simulated 6-axis IMU measurement is generated via measurement_model_2.
#   (d) These two sensor measurements (sensor 1: rough PCC position; sensor 2:
#       IMU measurement) are fused via the EKF to yield a refined position.
# =============================================================================
def main():
    # Simulation parameters
    num_steps = 50
    dt = 0.1  # time step in seconds

    # Generate a simulated trajectory for the true end-effector position (in mm)
    # For example, let the true position move along a circle in the XY-plane and vary in z.
    time = np.linspace(0, (num_steps-1)*dt, num_steps)
    radius = 30.0  # mm
    p_true_history = []
    for t in time:
        x_true = 30 + radius * np.cos(0.2 * t)  # offset by 30 mm in x
        y_true = 30 + radius * np.sin(0.2 * t)  # offset by 30 mm in y
        z_true = 100 + 5 * np.sin(0.1 * t)        # z varies slowly around 100 mm
        p_true_history.append(np.array([x_true, y_true, z_true]))
    p_true_history = np.array(p_true_history)  # shape (num_steps, 3)

    # For simulation, we assume the PCC measurement is obtained by passing the true p through constant_curvature.
    # In a real system, q would be measured by encoders.
    # Also, we compute the rough end-effector position as the centroid of the actuator positions.
    rough_meas_history = []
    for p in p_true_history:
        # Compute actuator measurements q (in mm)
        q = constant_curvature(p)
        # Compute rough position from q (in mm) and use the true z value
        rough_pos = rough_end_effector_position(q, p[2])
        rough_meas_history.append(rough_pos)
    rough_meas_history = np.array(rough_meas_history)  # shape (num_steps, 3)

    # Define sensor noise covariances (simulate measurement noise)
    R1_sim = np.eye(3) * 0.5    # Sensor 1 noise covariance (PCC rough measurement)
    R2_sim = np.eye(6) * 1.0    # Sensor 2 noise covariance (6-axis IMU)

    # For EKF initialization, we need sample measurements to compute covariances.
    # Here we generate 100 samples for each sensor based on a nominal true state.
    # We assume the true state (for initialization) is the first p_true in meters.
    x_nominal = (p_true_history[0] / 1000.0).reshape(3, 1)  # convert mm to meters for EKF state
    z1_samples = []
    z2_samples = []
    for _ in range(100):
        noise1 = np.random.multivariate_normal(np.zeros(3), R1_sim).reshape(3, 1)
        # For sensor 1, measurement model h1(x)=x; we simulate rough measurement from PCC.
        # Here, we use the nominal state (in meters) plus noise.
        z1_samples.append(x_nominal + noise1)
        noise2 = np.random.multivariate_normal(np.zeros(6), R2_sim).reshape(6, 1)
        # For sensor 2, we define h2(x) = [x; t*x] (with Rot=I, C2=0) and t = 0.1.
        z2_samples.append(np.vstack((x_nominal, 0.1 * x_nominal)) + noise2)
    z1_samples = np.hstack(z1_samples).T  # shape (100, 3)
    z2_samples = np.hstack(z2_samples).T  # shape (100, 6)

    # Define EKF measurement parameters
    # For sensor 1, we set Kf_1 = I and C_1 = 0.
    Kf_1 = np.eye(3)
    C_1 = np.zeros((3, 1))
    # For sensor 2, we set Kf_2 = I, C_2 = 0, Rot = I, and t = 0.1.
    Kf_2 = np.eye(3)
    C_2 = np.zeros((3, 1))
    Rot = np.eye(3)
    t_val = 0.1

    data = {
        'z_1': z1_samples,
        'z_2': z2_samples,
        'C_1': C_1,
        'C_2': C_2,
        'Kf_1': Kf_1,
        'Kf_2': Kf_2,
        'R': Rot,
        't': t_val
    }

    # Instantiate the EKF
    ekf = ExtendedKalmanFilter(data)

    # Store history (we convert positions to meters for EKF, while p_true is in mm)
    est_history = []
    true_history = []

    for i in range(num_steps):
        # For simulation, assume the "true" refined position (in meters) is p_true (converted to m)
        x_true = (p_true_history[i] / 1000.0).reshape(3, 1)
        
        # Sensor 1 measurement: rough end-effector position computed from PCC (convert mm to m)
        z1 = (rough_meas_history[i].reshape(3, 1)) / 1000.0
        # Add simulated noise
        noise1 = np.random.multivariate_normal(np.zeros(3), R1_sim).reshape(3, 1) / 1000.0
        z1 = z1 + noise1

        # Sensor 2 measurement: simulate IMU measurement using h2(x) = [x; t*x]
        noise2 = np.random.multivariate_normal(np.zeros(6), R2_sim).reshape(6, 1) / 1000.0
        z2 = measurement_model_2(x_true, C_2, Rot, t_val) + noise2

        # Stack measurements (9x1 vector)
        z_stack = np.vstack((z1, z2))
        
        # EKF prediction and correction
        ekf.prediction()
        ekf.correction_batch(z_stack)
        
        # Save histories
        est_history.append(ekf.x.flatten())
        true_history.append(x_true.flatten())

    est_history = np.array(est_history)
    true_history = np.array(true_history)
    
    print("Final true end-effector position (m):", true_history[-1])
    print("Final EKF estimated position (m):", est_history[-1])
    
    # Plot the evolution of the x, y, z coordinates over time
    plt.figure(figsize=(12, 8))
    labels = ['x', 'y', 'z']
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time, true_history[:, i], label=f'True {labels[i]}')
        plt.plot(time, est_history[:, i], '--', label=f'Estimated {labels[i]}')
        plt.ylabel(f'{labels[i]} (m)')
        plt.legend()
        plt.grid(True)
    plt.xlabel('Time (s)')
    plt.suptitle('EKF Fusion of PCC Rough Position and 6-Axis IMU for Precise End-Effector Position')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
