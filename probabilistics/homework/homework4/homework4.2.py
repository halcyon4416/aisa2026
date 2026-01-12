
# 모듈 로드
import pandas as pd
import numpy as np
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
import os

# 데이터 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "student_performance.csv")
data = pd.read_csv(data_file, encoding="UTF-8")
data = data.astype({"Performance Index": "int64"}) # 자료형이 올바르게 인식되도록 변경
data.columns = data.columns.str.replace(" ", "_") # 변수명이 공백 없이 인식되도록 변경

print("Data Head:")
print(data.head())
print("-" * 30)

# 2-1: 베이지안 선형회귀모형의 적합
# 1
tau_coef = 100

# 2
alpha_ig = 2
beta_ig = 0.5 

# 3
priors = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=tau_coef),
    "Hours_Studied": bmb.Prior("Normal", mu=0, sigma=tau_coef),
    "Previous_Scores": bmb.Prior("Normal", mu=0, sigma=tau_coef),
    "Extracurricular_Activities": bmb.Prior("Normal", mu=0, sigma=tau_coef),
    "sigma": bmb.Prior("InverseGamma", alpha=alpha_ig, beta=beta_ig)
}

# 4
model = bmb.Model(
    "Performance_Index ~ Hours_Studied + Previous_Scores + Extracurricular_Activities",
    data=data,
    priors=priors
)
idata = model.fit(draws=1000, chains=2, cores=1, random_seed=42)

# 5
print(az.summary(idata, var_names = ["Intercept", "Hours_Studied", "Previous_Scores", "Extracurricular_Activities"], hdi_prob = 0.95))

# 2-2: 베이지안 선형회귀모형의 해석
# a
summary = az.summary(idata, var_names=["Intercept", "Hours_Studied", "Previous_Scores", "Extracurricular_Activities"], hdi_prob=0.95)
possible_names = ["Extracurricular_Activities[T.Yes]", "Extracurricular_Activities[Yes]", "Extracurricular_Activities"]
var_name = None
for name in possible_names:
    if name in summary.index:
        var_name = name
        break

if var_name:
    hdi_lower = summary.loc[var_name, "hdi_2.5%"]
    hdi_upper = summary.loc[var_name, "hdi_97.5%"]
    print(f"a: [{hdi_lower:.4f}, {hdi_upper:.4f}]")

    # b
    # (0을 포함하지 않으므로 유의함)
    if (hdi_lower > 0) or (hdi_upper < 0):
        print("b: Significant")
    else:
        print("b: Not Significant")
else:
    print("Variable not found in summary")

