import numpy as np
from scipy.optimize import fsolve, root

def constant_curvature(p):
    x = p[0]
    y = p[1]
    z = p[2]
    d = 19  # Distance from the center of the chamber to the center of the base platform (mm)
    aa = 0  # Fixed distance from the base to the cross-section of the chamber

    # Special case: If the position is at the origin (0, 0), return straight lengths
    if x == 0 and y == 0:
        q1 = (z - aa) 
        q2 = (z - aa) 
        q3 = (z - aa)
        return np.array([q1, q2, q3])

    # Calculate the angle in the XY plane
    phi = np.arctan2(y, x)
    # Initial guess for the solver
    initial_guess = [x, y, z, 1.0, 0.1]  # Adjusted initial guess

    # Define the nonlinear equations to solve
    def nonlinear_equations(variables):
        x0, y0, z0, k0, theta0 = variables
        f = np.zeros(5)
        f[0] = 2 * np.sqrt(x0**2 + y0**2) / (x0**2 + y0**2 + z0**2) - k0
        f[1] = np.arccos(np.clip(1 - k0 * np.sqrt(x0**2 + y0**2), -1, 1)) - theta0
        f[2] = x0 + aa * np.sin(theta0) * np.cos(phi) - x
        f[3] = y0 + aa * np.sin(theta0) * np.sin(phi) - y
        f[4] = z0 + aa * np.cos(theta0) - z
        return f

    # Solve the nonlinear equations using fsolve
    try:
        solution = fsolve(nonlinear_equations, initial_guess, xtol=1e-12, maxfev=int(1e8))
    except Exception as e:
        print("Fsolve failed with exception:", e)
        return np.array([np.nan, np.nan, np.nan])

    # Extract real parts of the solution
    x1, y1, z1, k, theta = solution

    # Calculate the radii of curvature for each chamber
    R1 = 1 / k - d * np.sin(phi)
    R2 = 1 / k + d * np.sin(np.pi / 3 + phi)
    R3 = 1 / k - d * np.cos(np.pi / 6 + phi)

    # Calculate the length changes for each chamber
    q1 = float(theta * R1)
    q2 = float(theta * R2)
    q3 = float(theta * R3)

    return np.array([q1, q2, q3])

