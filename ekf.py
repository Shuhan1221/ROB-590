import numpy as np

# ---------------- PCC 正向运动学相关函数 ----------------

def compute_rotation_matrix(phi, beta):
    """
    根据 phi（方向角）和 beta（弯曲角）计算旋转矩阵。
    
    参数:
      phi : float - 方向角（弧度）
      beta: float - 弯曲角（弧度）
    
    返回:
      3x3 旋转矩阵（numpy 数组）
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
    根据曲率参数计算位移向量。
    
    参数:
      rho : float - 曲率
      beta: float - 弯曲角（弧度）
      phi : float - 方向角（弧度）
    
    返回:
      3D 位移向量（numpy 数组）
    """
    return (1 / rho) * np.array([(1 - np.cos(beta)) * np.sin(phi),
                                 (1 - np.cos(beta)) * np.cos(phi),
                                 np.sin(beta)])

def get_beta(l1, l2, l3, r):
    """
    根据三段长度计算弯曲角 beta。
    
    参数:
      l1, l2, l3 : float - 三段链路长度
      r          : float - 半径参数
      
    返回:
      beta (弧度)
    """
    return 2 * np.sqrt(l1**2 + l2**2 + l3**2 - l1*l2 - l1*l3 - l2*l3) / (3 * r)

def get_phi(l1, l2, l3):
    """
    根据三段长度计算方向角 phi。
    
    参数:
      l1, l2, l3 : float - 三段链路长度
      
    返回:
      phi (弧度)
    """
    return np.arctan2(3 * (l2 - l3), np.sqrt(3) * (l2 + l3 - 2 * l1))

def length_to_position_single(l, rigid_len, r):
    """
    计算单段末端的位置。
    
    参数:
      l         : list 或 numpy 数组 - 三段链路长度 [l1, l2, l3]
      rigid_len : float - 刚体部分的长度
      r         : float - 半径参数
      
    返回:
      3D 末端位置 (numpy 数组)
    """
    # 计算方向角和弯曲角
    phi = get_phi(l[0], l[1], l[2])
    beta = get_beta(l[0], l[1], l[2], r)
    lc = np.mean(l)  # 平均链路长度
    rho = beta / lc  # 计算曲率

    # 计算位移和旋转
    displacement = calc_displacement(rho, beta, phi)
    rotation = compute_rotation_matrix(phi, beta)
    direction = rotation @ np.array([0, 0, 1])  # 将 z 轴方向转换到末端实际方向

    # 末端位置 = 位移 + 刚体部分沿方向延伸
    end_position = displacement + direction * rigid_len
    return end_position

# ---------------- EKF 算法部分 ----------------

def f(mu_prev, u, rigid_len, r):
    """
    EKF 预测方程:
      mu_pred = f(mu_prev, u)
    其中:
      - [x,y,z] 由 u=[l1, l2, l3] 通过 PCC 正向运动学计算
      - [roll, pitch, yaw] 保持上一时刻状态（也可以根据实际模型更新）
    
    参数:
      mu_prev   : numpy 数组, 上一时刻状态 [x, y, z, roll, pitch, yaw]
      u         : numpy 数组, 控制量 [l1, l2, l3]
      rigid_len : float, 刚体部分的长度
      r         : float, PCC 模型中的半径参数
    
    返回:
      mu_pred   : numpy 数组, 预测状态
    """
    pos = length_to_position_single(u, rigid_len, r)
    # 直接继承上一时刻的姿态信息
    return np.hstack((pos, mu_prev[3:6]))

def h(mu):
    """
    EKF 中的观测方程:
      z_pred = h(mu)
    这里假设传感器直接测量状态中的姿态部分，即 [roll, pitch, yaw]
    
    参数:
      mu : numpy 数组, 状态向量
      
    返回:
      观测值向量
    """
    return mu[3:6]

def ekf_update(mu_prev, Sigma_prev, u, z, Q, R, rigid_len, r):
    """
    EKF 单步更新：结合预测和观测得到校正后的状态和协方差。
    
    参数:
      mu_prev   : 上一时刻状态均值 [x, y, z, roll, pitch, yaw]
      Sigma_prev: 上一时刻状态协方差 (6x6 矩阵)
      u         : 控制量 [l1, l2, l3]
      z         : 观测量 [roll_meas, pitch_meas, yaw_meas]
      Q         : 过程噪声协方差 (6x6 矩阵)
      R         : 观测噪声协方差 (3x3 矩阵)
      rigid_len : 刚体部分长度（用于 PCC 正向运动学）
      r         : PCC 模型参数（半径）
    
    返回:
      mu_new    : 更新后的状态均值
      Sigma_new : 更新后的状态协方差
    """
    # ----- 预测阶段 -----
    mu_pred = f(mu_prev, u, rigid_len, r)
    
    # 计算预测时状态转移函数 f 关于 mu_prev 的雅可比矩阵 F
    # 注意：由于 [x,y,z] 由 u 计算，与上一状态无关，因此对 mu_prev 求偏导为0；
    # 而姿态部分直接继承，因此对应部分为单位矩阵。
    F = np.zeros((6, 6))
    F[3:6, 3:6] = np.eye(3)
    
    # 外推协方差
    Sigma_pred = F @ Sigma_prev @ F.T + Q
    
    # ----- 更新阶段 -----
    # 计算预测观测
    z_pred = h(mu_pred)
    # 创新（观测残差）
    v = z - z_pred
    # 计算观测函数 h 关于 mu 的雅可比 H
    # 由于只量测姿态部分，对 [x,y,z] 的偏导为 0，对 [roll, pitch, yaw] 为单位矩阵
    H = np.zeros((3, 6))
    H[:, 3:6] = np.eye(3)
    
    # 创新协方差
    S = H @ Sigma_pred @ H.T + R
    # 卡尔曼增益
    K = Sigma_pred @ H.T @ np.linalg.inv(S)
    
    # 更新状态均值和协方差
    mu_new = mu_pred + K @ v
    Sigma_new = (np.eye(6) - K @ H) @ Sigma_pred
    # 为保证协方差矩阵的对称性，取其与转置的平均值
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
    
    return mu_new, Sigma_new

# ---------------- 示例运行 ----------------

if __name__ == "__main__":
    # 初始化状态：假设初始位置为原点，姿态为0
    mu = np.array([0, 0, 0, 0.0, 0.0, 0.0])  # [x, y, z, roll, pitch, yaw]
    Sigma = np.eye(6) * 0.01  # 初始协方差
    
    # 定义过程噪声和测量噪声协方差（可根据实际情况调参）
    Q = np.eye(6) * 0.001
    R = np.eye(3) * 0.01
    
    # 示例控制量：三条链路的长度（单位可以根据实际情况定义）
    l1, l2, l3 = 60.0, 80.0, 100.0
    u_k = np.array([l1, l2, l3])
    
    # PCC 正向运动学参数：刚体长度和半径参数（示例值）
    rigid_len = 0  # 刚体部分长度
    r = 19          # PCC 模型中的半径参数
    
    # 假设获得的观测（传感器测量的姿态）：单位为弧度
    z_k = np.array([0.1, 0.05, 0.0])
    
    # 执行一次 EKF 更新
    mu_updated, Sigma_updated = ekf_update(mu, Sigma, u_k, z_k, Q, R, rigid_len, r)
    
    print("更新后的状态 mu:")
    print(mu_updated)
    print("更新后的协方差 Sigma:")
    print(Sigma_updated)
