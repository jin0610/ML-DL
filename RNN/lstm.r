# 패키지 설치 및 불러오기
# install.packages("caret")
# install.packages("tibble")
# install.packages("forecast")
# install.packages("GGally")
# install.packages("ggplot2")
# install.packages("dplyr")

library(forecast)  # 시계열 예측 관련 패키지(ARIMA, 지수평활 등) - auto.arima(), ets(), forecast() 등)
library(stats)     # 통계 분석을 위한 패키지(기술통계량, 가설 검정, 회귀분석 등) - mean(), sd(), anova() 등
library(tseries)   # 시계열 데이터 분석 관련 패키지 - adf.test(), kpss.test(), diff(), arima(), acf(), pacf() 등
library(ggplot2)   # 데이터 시각화 도구
library(GGally)    # ggplot2 패키지 기반 시각화 도구 - ggpairs(), ggduo(), ggmatrix() 등
library(dplyr)     # 데이터 조작 및 전처리 도구 - select(), filter(), summarize(), %>% 등
library(caret)     # 머신러닝 모델 구축 및 평가 관련 도구
library(seasonal)  # 계절성 분석 수행 도구 - seas(), seasonal(), irregulatr(), des()
library(zoo)       # 시계열 데이터 다루는 도구 (시간에 따라 정렬된 데이터 처리, 조작 도구)
library(keras)     # 딥러닝 모델 구축, 훈련 패키지
library(tensorflow)# 딥러닝, 기계 학습을 위한 오픈 소스 라이브러리

##### 1. 데이터 불러오기
data <- read.csv('./data/효령노인복지타운.csv', 
                 na.strings = ".", 
                 header = TRUE, encoding = "utf-8")

##### 2. 데이터 탐색 및 전처리
View(data) # 데이터 보기
str(data)  # 데이터 속성

### 2-1. 필요하지 않은 데이터 삭제
data<- data[-c(2,4,6,8,10,12)]
names(data) <- c('date','pm10', 'pm25', 'o3', 'no2','co','so2')

# 2020-12 데이터 제거
data$date <- as.POSIXct(data$date, format = "%Y-%m-%d %H:%M")
str(data)

data <- data[data$date >= "2021-01-01 00:00",]

View(data)
str(data)

# # 추세(선형 그래프)
# plot(data$pm10,type = 'l')
# plot(data$pm25,type = 'l')
# plot(data$o3,type = 'l')
# plot(data$no2,type = 'l')
# plot(data$co,type = 'l')
# plot(data$so2,type = 'l')

# 주기성
autoplot(decompose(data$pm10))
autoplot(decompose(data$pm25))
autoplot(decompose(data$o3))
autoplot(decompose(data$no2))
autoplot(decompose(data$co))
autoplot(decompose(data$so2))

# 상관 관계
#data_nan <- na.omit(data)
#ggpairs(data_nan)

# 결측치 확인 및 처리 - 선형 보간
table(is.na(data))

data$pm10 <- na.approx(data$pm10)
data$pm25 <- na.approx(data$pm25)
data$o3 <- na.approx(data$o3)
data$no2 <- na.approx(data$no2)
data$co <- na.approx(data$co)
data$so2 <- na.approx(data$so2)


### 2-2. 데이터 정규화

# x, y 나누기
x <- data %>% select(pm10, pm25, o3, no2, co, so2 )
y <- data %>% select(pm10)

# MinMax Scaling
scaled_x <- preProcess(x, method = "range")
scaled_y <- preProcess(y, method = "range")

# 스케일링 적용
scaled_x <- predict(scaled_x, newdata = x)
scaled_y <- predict(scaled_y, newdata = y)
scaled_y_2 <- scale(y, center=min(y), scale = max(y) - min(y))
scaled_y_2 <- as.data.frame(scaled_y_2)

str(scaled_x)
str(scaled_y)
cat(dim(scaled_x), '\n', dim(scaled_y))


################  변경 필요 부분  #################################################################
### 2-3. 새로운 차원 추가 -> 3차원 데이터셋 생성
# 최종 형태 : ((21132, 12, 6), (21132, 1))
time_steps <- 12

x_matrix <- as.matrix(scaled_x)
y_matrix <- as.matrix(scaled_y)

cat(dim(x_matrix), '\n', dim(y_matrix))

x_3d <- t(sapply(1:(nrow(x_matrix) - time_steps),
                         function(x) x_matrix[x:(x + time_steps - 1), ]))
y_3d <- t(sapply(1:(length(y_matrix) - time_steps),
                         function(y) y_matrix[(y + time_steps), ]))
dim(y_3d)
dim(y_matrix)
x_data <- array(
  data = as.numeric(unlist(x_3d)),
  dim = c(
    nrow(x_3d),
    time_steps,
    dim(scaled_x)[2]))

y_data <- array(data = as.numeric(unlist(y_3d)),
                dim = c(dim(y_3d)[2],1))

dim(x_data)
dim(y_data)
##### 3. 학습 데이터 분리

# 전체 데이터셋을 학습 데이터셋과 나머지 데이터셋으로 분리
train_ratio <- 0.8
train_idx <- round(nrow(x_data) * train_ratio,0)

X_train <- x_data[1:train_idx, , ]
x_test <- x_data[-(1:train_idx), , ]
y_train <- y_data[1:train_idx, ]
y_test <- y_data[-(1:train_idx), ]

dim(X_train)
dim(x_test)

dim(y_train)
dim(y_test)

# 나머지 데이터셋을 검증 데이터셋과 평가 데이터셋으로 분리
val_ratio <- 0.5
val_idx <- round(nrow(x_test) * val_ratio,0)

X_val <- x_test[1:val_idx, , ]
x_test <- x_test[-(1:val_idx), , ]
y_val <- y_test[1:val_idx ]
y_test <- y_test[-(1:val_idx)]


###################################################################################

##### 4. 모델 생성 및 학습
model <- keras_model_sequential()
model %>%
  layer_lstm(units = 50, input_shape = c(dim(X_train)[2], dim(X_train)[3]), return_sequences = TRUE) %>%
  layer_dropout(0.5) %>%
  layer_lstm(units = 50) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 1)

model %>% compile(loss = 'mse', optimizer = 'adam', metrics = c('mse'))

early_stop <- callback_early_stopping(monitor = 'val_loss', patience = 10)

summary(model)

##### 5. 모델 평가 (mae/mse)

h1 <- model %>% fit(X_train, y_train,
                    validation_data = list(X_val, y_val),
                    epochs = 100)

lstm_forecast <- model %>% predict(x_test)
lstm_forecast_unscaled <- lstm_forecast * min(y) * max(y) + min(y)

result <- data.frame()
df <- data.frame(actual = y_test, pred = lstm_forecast)

ggplot(df) +
  geom_line(aes(x = 1:nrow(df), y = actual), color = "blue", linetype = "solid", linewidth = 1) +
  geom_line(aes(x = 1:nrow(df), y = pred), color = "red", linetype = "dashed", linewidth = 1) +
  xlab("Index") +
  ylab("Value") +
  ggtitle("Actual vs. Predicted") +
  theme_minimal()
