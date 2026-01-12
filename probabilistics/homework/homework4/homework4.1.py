
# 모듈 로드
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

# 데이터 로드
# 스크립트 파일과 동일한 디렉토리에 있는 student_performance.csv 파일을 로드합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "student_performance.csv")
data = pd.read_csv(data_file, encoding="UTF-8")

# 전처리
# 자료형 변환 및 컬럼명 변경
data = data.astype({"Performance Index": "int64"})
data.columns = data.columns.str.replace(" ", "_")

print("Data Head:")
print(data.head())
print("-" * 30)

# 1-1: 범주형 변수를 포함한 선형회귀모형의 적합
# 1
model = smf.ols(formula='Performance_Index ~ Hours_Studied + Previous_Scores + Extracurricular_Activities', data=data)
model_fit = model.fit()

print(model_fit.summary())

# 1-2: 범주형 변수를 포함한 선형회귀모형의 적합도 검정
# a 
f_pvalue = model_fit.f_pvalue
print(f"a: {f_pvalue:.4e}")
# (p-value < 0.05 이므로 유의함)

# b
print("b:")
print(model_fit.pvalues)
# (모든 변수의 p-value < 0.05 이므로 유의함)

# c
r_squared = model_fit.rsquared
adj_r_squared = model_fit.rsquared_adj
print(f"c: R2={r_squared:.4f}, Adj-R2={adj_r_squared:.4f}")

# 1-3: 범주형 변수를 포함한 선형회귀모형의 해석
params = model_fit.params
intercept = params['Intercept']
coef_hours = params['Hours_Studied']
coef_prev = params['Previous_Scores']
if 'Extracurricular_Activities[T.Yes]' in params:
    coef_extra = params['Extracurricular_Activities[T.Yes]']
else:
    coef_extra = 0

# a
print(f"a: Performance_Index = {intercept + coef_extra:.4f} + {coef_hours:.4f} * Hours_Studied + {coef_prev:.4f} * Previous_Scores")

# b
print(f"b: Performance_Index = {intercept:.4f} + {coef_hours:.4f} * Hours_Studied + {coef_prev:.4f} * Previous_Scores")

# c
print(f"c: {coef_extra:.4f}")

