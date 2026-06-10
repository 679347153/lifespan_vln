import math

def calculate_exponential_weights(N, a=0.1, b=0.25):
    """
    使用指数函数计算动态权重。

    参数:
    N (float): 当前时间。
    a (float): N=0时的初始值。
    b (float): 增长率。b越大，增长越快。

    返回:
    tuple: (omega_N1, omega_N2)
    """
    # 计算 omega_N2，其值从初始值a开始指数增长
    omega_N2_raw = a * math.exp(b * N)

    # 为了避免权重超过1，我们将其限制在[0, 1]范围内
    omega_N2 = max(0, min(1, omega_N2_raw))
    
    # omega_N1 = 1 - omega_N2，其值从初始值1-a开始指数衰减
    omega_N1 = 1 - omega_N2
    
    return omega_N1, omega_N2

def calculate_sigmoid_weights(N, k=1.0, N0=5.0):
    """
    使用 Sigmoid 函数计算动态权重。
    
    参数:
    t (float): 当前时间。
    k (float): 曲线的陡峭程度。k越大，变化越快。
    N0 (float): 曲线的中心点。t = N0 时，\\omega_N2 为 0.5。

    返回:
    tuple: (omega_N1, omega_N2)
    """
    # 计算 omega_N2，其值从接近0平滑地增加到接近1
    omega_N2 = 1 / (1 + math.exp(-k * (N - N0)))

    # omega_N1 = 1 - omega_N2，其值从接近1平滑地减少到接近0
    omega_N1 = 1 - omega_N2
    
    return omega_N1, omega_N2
