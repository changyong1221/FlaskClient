from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

# differential privacy
# 差分隐私配置参数
q = 0.03
eps = 8.0
delta = 1e-5
tot_T = 100
E = 1
sigma = compute_noise(1, q, eps, E*tot_T, delta, 1e-5)      # 高斯分布系数
