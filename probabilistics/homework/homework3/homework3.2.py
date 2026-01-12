import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm

# true theta value(true theta의 값은 사실 unknown이지만 편의를 위해 theta = 3에서 데이터를 추출하자)
theta_true = 3

# 시드 고정
np.random.seed(1004)

# 관측치 생성
n = 100
obs = norm.rvs(loc = theta_true, scale = 1, size = n)
x_bar = np.mean(obs)

# nu 값 설정
nu = 5

# 2-1: Metropolis-Hastings Algorithm 코드 작성 및 결과 시각화
# 1
theta_init = x_bar

# 2
T = 10000

# 3
theta_sample = []

# 4
theta_prev = theta_init

# 5
for t in range(T):
    # 제안분포에서 후보 Y 샘플링: Y ~ N(x_bar, 1/n)
    Y = norm.rvs(loc=x_bar, scale=np.sqrt(1/n))
    
    # acceptance ratio 계산
    # alpha(theta^(t-1), Y) = min(1, ratio)
    # ratio = posterior(Y) / posterior(theta_prev)
    # posterior ∝ (1 + Y^2/nu)^(-(nu+1)/2) / (1 + theta_prev^2/nu)^(-(nu+1)/2)
    
    numerator = (1 + Y**2 / nu) ** (-(nu + 1) / 2)
    denominator = (1 + theta_prev**2 / nu) ** (-(nu + 1) / 2)
    ratio = numerator / denominator
    alpha = min(1, ratio)
    
    # acceptance 여부 결정
    u = uniform.rvs()
    if u < alpha:
        theta_curr = Y
    else:
        theta_curr = theta_prev
    
    theta_sample.append(theta_curr)
    
    # 현재 값을 이전 값으로 취급 (다음 step 업데이트 진행 위해)
    theta_prev = theta_curr

# 6
burn_cnt = 5000
theta_sample_b = theta_sample[burn_cnt:]

# visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# hist
axes[0].hist(theta_sample, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[0].set_xlabel('theta')
axes[0].set_ylabel('Density')
axes[0].set_title('Histogram of theta (after burn-in)')
axes[0].grid(True, alpha=0.3)

# traceplot
axes[1].plot(theta_sample_b, linewidth=0.5)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('theta')
axes[1].set_title('Trace plot of theta (after burn-in)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"total: {T}")
print(f"after burn-in: {len(theta_sample_b)}")
print(f"x_bar: {x_bar:.4f}")

# 2-2
# a
theta_posterior_mean = np.mean(theta_sample_b)
print(f"(a) $\theta$에 대한 posterior mean: {theta_posterior_mean:.4f}")

# b
theta_posterior_median = np.median(theta_sample_b)
print(f"(b) $\theta$에 대한 posterior median: {theta_posterior_median:.4f}")

# c
theta_posterior_var = np.var(theta_sample_b, ddof=1)
print(f"(c) $\theta$에 대한 posterior variance: {theta_posterior_var:.6f}")

# d
theta_ci_L = np.percentile(theta_sample_b, 2.5)
theta_ci_U = np.percentile(theta_sample_b, 97.5)
print(f"(d) $\theta$에 대한 95% credible interval: [{theta_ci_L:.4f}, {theta_ci_U:.4f}]")