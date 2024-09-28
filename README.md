# 혼합모형

#일시
- 2022-09 ~ 2202-12

# 주제
- 서울시(여의도, 상암 지역) 공공자전거 사용자 분류 및 운동량 예측

# 배경 및 목적
- 서울시에는 공공자전거 이용자들의 이용 목적은 다양
- 여의도와 상암지역에 있는 공공자전거 이용자들을 GMM과 MoE를 활용하여 이용자들을 분류하고 이들의 운동량을 예측
 

# 데이터
- 서울시공공데이터(www.data.seoul.go.kr) 에 있는 서울시 공공자전거 이용정보(시간대별) 데이터
- 데이터 개수 : 3704328개
- USE_CNT(이용건수), EXER_AMT(운동량), CARBON_AMT(탄소 절감량), MOVE_METER(이동거리)라는 4개의 수치형 변수와 RENT_DT(대여일자), RENT_ID(대여소번호), RENT_NM(대여소명), RENT_HR(대여시간), RENT_TYPE(대여구분코드), GENDER_CD(성별), AGE_TYPE(나이), , MOVE_TIME(이동시간)라는 8개의 범주형 변수로 구성
  

# 사용언어/모델
- R/ GMM(Gaussian Mixture Model) MoE(Mixtuer of Experts)

# 모델 성능 지표
- BIC, RMSE

# 데이터 전처리
- 6월 1일부터 6월 8일 까지의 총 2000개 데이터만 활용(비가 오지 않으면서 지방선거 날과 현충일이 있는 등 8일동안 평일과 휴일이 적절히 잘 섞여 있는 기간이기에 이 기간으로 설정)
- ‘대여시간’ 변수의 범주는 0시부터 23시까지 총 24개의 범주로 이루어져 있다. 0시부터 06시 사이의 이용자수들은 다른 시간대에 비해 상대적으로 이용자수가 적기 때문에 이 시간대 이용자들은 분석 대상에 미포함(3시간 간격으로 총 6개범주로 처리).
- AGE_TYPE 변수는 60대 70대는 50대와 통합
- USE_CNT 변수는 이용건수가 1번인 이용자 1, 1번 이상인 이용자는 0으로 처리
- USE_TYPE 변수는  정기권은 1, 정기권이 아닌 이용권들은 0으로 처리
- FEATURE ENGINEERING: RENT_SPOT(대여장소)(업무,지하철,주거), park(공원 인접 여부), holiday(휴일 여부)

# 프로젝트 내용
- BIC 기준(k=2일때 -10657.99, k=3일때 -10122.29, k=4일때 -9835.712) component가 2인 GMM 모형 fitting /추정 latent variable: 속도, 긴박성
- BIC 기준(k=3일때 -155586.71, k=4일때 -14672.784, k=5일때 -14601.217) component가 3인 MoE 모형 fitting / 추정latent variable: riding purpose, willpower(desire) of riding
- MoE 모형 으로 나온 cluster 별 평균 운동량 예측(cluster1: 83.01738, cluster2: 80.8889, cluster3: 191.2407)
- 운동량의 RMSE와 MAE를 기준으로 MoE를 RANDOM FOREST와 LINEAR MODEL과 비교
- RMSE(randomforest: 61.07248, linear model: 30.85834, MoE: 27.87566) -> MoE가 가장 좋은 성능 보임


# 기대효과
- 서울시(여의도, 상암지역)의 공공자전거 이용자 분류를 MoE를 통해 분류하여 이들이 어떤 component에 속할지 예측.  
- 얼마만큼의 운동량을 가지는지 예측.
