# Note

## Lecture 1

기대값: balance point라고도 부름.
중심: 정의하기 나름.
중앙값: 좌,우 0.5로 구분하는 지점.
최빈값: 확률이 가장 높은 지점.

1차 적률: E(X)
2차 적률: E(X^2) X^2 = Y = E(Y), 퍼져있는 정도
3차 적률: E(X^3) skewness (왜곡도, 좌우대칭) 정규분포: 0
4차 적률: E(X^4) 첨도(kurtosis)(꼬리두께) 정규분포: 3

"모든 적률이 동일하면 두 분포는 같다"
  also: MGF (Moment Generating Function)이 같다면 두 분포는 같다.

기대값 mean 가장 흔한 방법 = sample mean

분산/표준편차
Spread. 변동성을 나타낸다. >=0
분산
  : E(x)-u = 0이 되기 때문에 square로 변환했다.

표준편차
  : 분산에 square를 씌울 경우 단위가 변해. m -> m^2
  그렇기 때문에 square root를 취하여 표준편차를 정의한다.

---
---

## Lecture 2

1. 베르누이.
  세타(확률)만 알면 확률을 구할 수 있다.
  간단하지만 세타(에러율 성공률등) 모델링하여 설명하는 것이 관건
  0,1 분포를 따르는 다른 분포와 결합: logistic regression

2. 이항분포.
  베르누이를 여러번 반복한 결과를 나타낸다.
  
3. 포아송분포.
  평균과 분산이 같다. = 모든 사건이 독립적이다.
  데이터를 분석했는데 평균과 분산이 많이 다르다면 독립적인 사건이 아니라는 걸 알 수 있다.
  모수: lambda 항상 양수
  log 회귀분석을 통해 다른 분포와 결합시킬 수 있다.
  example story: flying bomb 

---

4. 베타분포
  alpha, beta를 구하자.
  모수적, 비모수적 분포 분석
  대표적 비모수적 분포 분석: 히스토그램.
  감마함수 B(a,b) -> 적분해서 1 만들어주는 정규화 상수

5. 지수분포
  람다, 로로 표현식이 다양함. 밑에 붙는 숫자를 scale param
  독립일 경우 memoryless property를 만족한다.
    -> 이를 특징으로 가지는 분포는 지수분포만 있다.

6. 정규분포
  선형변환을 해도 정규분포 폼을 유지함.
  outlier가 많은 분포에서는 정규분포를 사용하지 않는 것이 좋다.

percentile 좀 더 일반적으로 quantile(분위수), quartile(사분위수)

---

결합확류분포, 마지널, 조인트

조인트 알면 마지널 구할 수 있음.
마지널 알면 독립인 가정하에 조인트를알 수 있음.
 마지널로 조인트 구하는 방법이 있음(코퓰러?)

covariance:
기호는 sigma(x,y)로 보통 쓰임.

두 분포의 값이 같은 방향으로 움직일 때 양수.
다른 방향으로 움직일 때는 음수.
E(XY) - E(X)E(Y) Y를 X로 치환화면 그냥 분산이 된다.
cov matrix : symetric matrix. 
  cov(x,y), cov(y,x)는 같다. -> 순서 바뀐다고 관계가 달라지지 않아.

inv cov matrix: precision matrix

---

Correlation coefficient

measurement의 단위가 변하는 등의 상황에서. cov는 이런 상황 대응이 안된다.
이 때, 각각의 std dev로 나눠주면 상관관계를 구할 수 있다. unit free한 값이 된다.
-1~1 사이의 값을 가짐.
상관계수의 경우 선형관계가 아닌 관계를 표현하는데는 적합하지 않다.
Correlation. 선형관계일 경우 1

3페이지 3번째식 (x+y)^2 전개식 형태임.
두 데이터가 관계가 없을 경우 1차항이 0이됨. == cov = 0

독립이면 관련이 없다. cov=0일 때 관련이 없는데 독립이 아닐 수 있다.

---

Anscombe’s quartet

 x,y 쌍 다 평균 분산 correlation이 같음. 소수점 4째자리까지.
 lesson: 숫자만 가지고 판단하지 말고 그림을 그려보자.
 선형이 아닌 케이스 (2) 
 outlier (3,4)

---
 
확률변수(데이터)의 "독립"

모든 x,y,...에 대해서 P(X=x, Y=y,...) = P(X=x)P(Y=y,...)가 성립할 때

Var(X+-Y) = Var(X) + Var(Y) 분산은 무조건 커진다 개념으로 이해하자.

---

조건부 확률분포

- P(X=x|Y=y) = P(X=x, Y=y) / P(Y=y)
 -> Y가 주어졌을 때 X의 확률분포(x가 움직임)

  - 집합표현  
    P(A|B) = P(A ∩ B) / P(B)

 - 48p
  exp아래쪽항은 적분해서 1만드는 정규화 상수
  positive-definite matrix 여기엔 어떤 벡터를 갖고와도 양수가 나온다

    - x^T * Σ * x : Quadratic form

- cholesky decomposition
  : positive-definite matrix를 lower triangular matrix로 분해하는 방법
    positive-definite matrix에서만 적용 가능

- cov matrix의 원소가 0이고 분포가 정규분포일 경우 원소 i,j 는 독립이다.

- 50p. 곱해진 각 w는 wieght임.(양수 only, 적분시 1이어야 함.)

  discrete인데 혼합분포(bernoulli)를 사용할 수 있는 대표적 예: mnist

---

## Lecture 3

