# 과제3 답안

## 문제 1
### **[ 문제 1-(1) ]** Gibbs Sampling 코드 작성 및 결과 시각화
![Figure_3-1](Figure_3-1.png)
```python

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
```

### **[ 문제 1-(2) ]** theta에 대하여 다음의 값들을 구하여라.
#### (a) theta에 대한 posterior mean
> 답: 0.7133

#### (b) theta에 대한 posterior median
> 답: 0.7213

#### (c) theta에 대한 posterior variance
> 답: 0.009292

#### (d) theta에 대한 95% credible interval
> 답: [0.5091, 0.8811]

```python

# a
theta_posterior_mean = np.mean(theta_sample_b)
print(f"(a) theta에 대한 posterior mean: {theta_posterior_mean:.4f}")

# b
theta_posterior_median = np.median(theta_sample_b)
print(f"(b) theta에 대한 posterior median: {theta_posterior_median:.4f}")

# c
theta_posterior_var = np.var(theta_sample_b, ddof=1)
print(f"(c) theta에 대한 posterior variance: {theta_posterior_var:.6f}")

# d
theta_ci_L = np.percentile(theta_sample_b, 2.5)
theta_ci_U = np.percentile(theta_sample_b, 97.5)
print(f"(d) theta에 대한 95% credible interval: [{theta_ci_L:.4f}, {theta_ci_U:.4f}]")

```

### **[ 문제 1-(3) ]** x_20에 대하여 다음의 값들을 구하여라.

#### (a) x_20에 대한 posterior mean
> 답: 0.7110

#### (b) x_20에 대한 posterior variance
> 답: 0.205520

```python

# a
x20_posterior_mean = np.mean(x20_samples_b)
print(f"(a) x_20에 대한 posterior mean: {x20_posterior_mean:.4f}")

# b
x20_posterior_var = np.var(x20_samples_b, ddof=1)
print(f"(b) x_20에 대한 posterior variance: {x20_posterior_var:.6f}")

```

---
## 문제 2

### **[ 문제 2-(1) ]** Metropolis-Hastings Algorithm 코드 작성 및 결과 시각화

![Figure_2-1](Figure_3-2.png)
```python

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


```

### **[ 문제 2-(2) ]** theta에 대하여 다음의 값들을 구하여라.
#### (a) theta에 대한 posterior mean
> 답: 2.9907

#### (b) theta에 대한 posterior median
> 답: 2.9904

#### (c) theta에 대한 posterior variance
> 답: 0.010081

#### (d) theta에 대한 95% credible interval
> 답: [2.7956, 3.1866]

```python

# a
theta_posterior_mean = np.mean(theta_sample_b)
print(f"(a) theta에 대한 posterior mean: {theta_posterior_mean:.4f}")

# b
theta_posterior_median = np.median(theta_sample_b)
print(f"(b) theta에 대한 posterior median: {theta_posterior_median:.4f}")

# c
theta_posterior_var = np.var(theta_sample_b, ddof=1)
print(f"(c) theta에 대한 posterior variance: {theta_posterior_var:.6f}")

# d
theta_ci_L = np.percentile(theta_sample_b, 2.5)
theta_ci_U = np.percentile(theta_sample_b, 97.5)
print(f"(d) theta에 대한 95% credible interval: [{theta_ci_L:.4f}, {theta_ci_U:.4f}]")

```