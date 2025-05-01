import numpy as np
import gtsam
from gtsam import Rot3, NonlinearFactorGraph, Values, BetweenFactorRot3, PriorFactorRot3
from gtsam.symbol_shorthand import X  # X(0), X(1), ...

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


def compute_rot3(phi, beta):
    R = compute_rotation_matrix(phi, beta)       
    return Rot3(R)


u_k    = np.array([10.0, 50.0, 10.0])            # control input
phi_k  = u_k[0] / (np.sum(u_k) + 1e-8)          
beta_k = np.linalg.norm(u_k) / 100.0            
z_k    = np.array([0.1, 0.05, 0.2])              # IMU‐measured [roll,pitch,yaw]

# Noise covariances from your Q,R
Q  = np.eye(3) * 0.001      
R  = np.eye(3) * 0.01       
# GTSAM wants σ (std dev), so take √diag
proc_sigmas = np.sqrt(np.diag(Q))
meas_sigmas = np.sqrt(np.diag(R))

process_noise = gtsam.noiseModel.Diagonal.Sigmas(proc_sigmas)
meas_noise    = gtsam.noiseModel.Diagonal.Sigmas(meas_sigmas)

# --- Build graph & initial guesses ---
graph  = NonlinearFactorGraph()
initial = Values()

# 1) Prior on X(0) = identity rotation
rot0 = Rot3.Ypr(0.0, 0.0, 0.0)
prior_noise = gtsam.noiseModel.Diagonal.Sigmas([1e-3, 1e-3, 1e-3])
graph.add(PriorFactorRot3(X(0), rot0, prior_noise))
initial.insert(X(0), rot0)

# 2) Between‐factor from X(0)→X(1) using your control model
ΔR = compute_rot3(phi_k, beta_k)
graph.add(BetweenFactorRot3(X(0), X(1), ΔR, process_noise))
initial.insert(X(1), rot0.compose(ΔR))

# 3) “Measurement” factor on X(1) from IMU
#    Here we treat the IMU euler‐angles z_k as a rotation prior
rot_meas = Rot3.Ypr(z_k[2], z_k[1], z_k[0])
graph.add(PriorFactorRot3(X(1), rot_meas, meas_noise))

# --- Optimize ---
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
result    = optimizer.optimize()

# --- Extract and print the updated orientation estimate ---
opt_rot1 = result.atRot3(X(1))
roll, pitch, yaw = opt_rot1.roll(), opt_rot1.pitch(), opt_rot1.yaw()
print(f"Optimized orientation (r,p,y): {roll:.4f}, {pitch:.4f}, {yaw:.4f}")
