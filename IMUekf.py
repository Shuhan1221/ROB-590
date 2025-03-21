def process_model()

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
