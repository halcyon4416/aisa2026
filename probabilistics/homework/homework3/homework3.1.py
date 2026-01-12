import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, bernoulli

# data for the 19 observed individuals
obs = np.array([1,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,1])

# y is defined as the sum of 19 observed values
y = obs.sum()

# 1-1: Gibbs Sampling 코드 작성 및 결과 시각화
# 1
theta_init = 0.5
x_20_init = 1

# 2
T = 10000

# 3
theta_sample = []
x20_samples = []

# 4
theta_prev = theta_init
x20_prev = x_20_init

# 5
for t in range(T):
    # update x20
    x20_curr = bernoulli.rvs(p=theta_prev)
    
    # update theta
    alpha = y + x20_curr + 1
    beta_param = 20 - y - x20_curr + 1
    theta_curr = beta.rvs(a=alpha, b=beta_param)
    
    # 배열에 추가
    theta_sample.append(theta_curr)
    x20_samples.append(x20_curr)
    
    # 현재 값을 이전 값으로 취급 (다음 step 업데이트 진행 위해)
    theta_prev = theta_curr
    x20_prev = x20_curr

# 6
burn_cnt = 5000
theta_sample_b = theta_sample[burn_cnt:]
x20_samples_b = x20_samples[burn_cnt:]

# visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# hist
axes[0].hist(theta_sample_b, bins=50, density=True, alpha=0.7, edgecolor='black')
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
print(f"y: {y}")

# 1-2
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

# 1-3
# a
x20_posterior_mean = np.mean(x20_samples_b)
print(f"(a) $x_{20}$에 대한 posterior mean: {x20_posterior_mean:.4f}")

# b
x20_posterior_var = np.var(x20_samples_b, ddof=1)
print(f"(b) $x_{20}$에 대한 posterior variance: {x20_posterior_var:.6f}")
