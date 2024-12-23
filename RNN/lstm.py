#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
from  sklearn.metrics  import mean_squared_error
from  sklearn.metrics  import mean_absolute_error

# ## 데이터 불러오기 및 통합
data = pd.read_csv("./효령노인복지타운.csv", encoding = 'utf-8', index_col='dateInfo', parse_dates=True)

# ## 데이터 탐색 및 전처리
# 데이터 크기 확인
data.shape

# 데이터 컬럼명 확인
data.columns

# 데이터 확인
data.head()

# 필요 데이터 추출
data = data[['pm10Score', 'pm25Score', 'o3Score', 'no2Score', 'coScore', 'so2Score']]

# 데이터 컬럼명 변경
data.columns = ['pm10', 'pm25','o3','no2','co','so2']

# 2020년 12월 데이터 삭제
data = data['2021-01-01 00:00:00':]

# 데이터 기초 정보(데이터 개수, 평균, 분산 최소값 등) 확인
data.describe()

# 데이터 타입 확인
data.dtypes


# 데이터 결측값 확인 및 결측치 처리
print(data.isnull().sum())

# 결측치 처리 - 선형 보간
data = data.interpolate()
data

# ### 변수 간 상관관계 파악
plt.figure(figsize=(16,9))
sns.heatmap(data.corr(), annot=True,linewidths=0.5)


# ## 정규화(스케일링)
scaler = MinMaxScaler()

x_cols = ['pm10', 'pm25', 'o3', 'no2', 'co', 'so2']
y_cols = ['pm10']

scaler_x = scaler.fit_transform(data[x_cols])
scaler_y = scaler.fit_transform(data[y_cols])

# ## 3D 형태로 데이터 변경
window_size = 24

x = []
y = []

for i in range(window_size, scaler_x.shape[0]):
    x.append(scaler_x[i-window_size : i])
    y.append(scaler_y[i])
    
x, y = np.array(x), np.array(y)
x.shape, y.shape


# ## 학습데이터 분리
# 전체 데이터셋을 학습 데이터셋과 나머지 데이터셋으로 분리
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=False) # train set 60%, test set : 40%

# 나머지 데이터셋을 검증 데이터셋과 평가 데이터셋으로 분리
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)

print(X_train.shape, X_val.shape, X_test.shape)

# lstm 모델 생성
model = Sequential()
model.add(LSTM(units = 50, input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(units = 50))
model.add(Dropout(rate=0.5))
model.add(Dense(1)) 

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# model의 성능 지표가 설정한 epoch 동안 개선되지 않을 때 조기종료
early_stop = EarlyStopping(monitor='val_loss', 
                           patience=10)  # 10회동안 개선되지 않으면 조기종료 

print(model.summary())

# 모델 학습
h1 = model.fit(X_train,y_train,
                  validation_data=(X_val,y_val),
                  epochs=100)


# 성능 및 과적합 여부 확인 (train data의 mse와 val data의 mse 비교)
plt.figure(figsize=(15,5))
plt.plot(h1.history['loss'], label='loss')
plt.plot(h1.history['val_loss'], label='val_loss')
plt.legend()
plt.grid()
plt.show()


# ## 모델 test
result=pd.DataFrame()
result['pm10'] = y_test.reshape(X_test.shape[0])

# test 데이터를 이용하여 예측
result['pre'] = model.predict(X_test)

# 실제값 비교하기 위한 역정규화 
result[['pm10','pre']] = scaler.inverse_transform(result[['pm10','pre']])

# 비교 그래프 그리기
plt.figure(figsize=(15,5))
plt.plot(result['pm10'][1000:1500], label='actual')
plt.plot(result['pre'][1000:1500], label='pred')
plt.legend()
plt.grid()
plt.show()


# ## 모델 평가

mse1 = mean_squared_error(result['pm10'], result['pre'])
print('mse  = {:.3f}'.format(mse1))

mae1 = mean_absolute_error(result['pm10'], result['pre'])
print('mse  = {:.3f}'.format(mae1))



