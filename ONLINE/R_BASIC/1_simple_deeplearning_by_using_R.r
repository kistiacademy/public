
# path setting
currentPath = "./"
dataPath = "data/"
figPath = ".figures/"
modelPath = "models/"

dataPath <- paste(currentPath, dataPath, sep="")
figPath <- paste(currentPath, figPath, sep="")

# 전체 data 중 training에 80%, testing에 20%에 사용.
seperatingRate = 0.8
# 학습 데이터량에 따른 성능 비교를 위해 먼저 50%의 데이터만 사용.
maxSizeRate = 0.5

# 변수 초기화 후 데이터 로딩
fileName = "oqmd-simple.csv"
all_data = 0
all_data = read.csv(paste(dataPath,fileName, sep=""), sep=",", header=TRUE)
attach(all_data)

# 데이터 확인
all_data

# 총 데이터의 수 (rows)
nrow(all_data)

# 총 입출력 레이블의 수 (columns)
length(all_data)

# 함수 선언: NaN data to zero
na.zero <- function (x) {
    x[is.na(x)] <- 0
    return(x)
}

# 데이터에 적용
all_data <- na.zero(all_data)

# down scaling
seperatingValue = 0
seperatingValue = floor(nrow(all_data)*maxSizeRate*seperatingRate)

# train/test data 분리
set.seed(1234)
all_data_reduced = all_data[sample(nrow(all_data)*maxSizeRate), ]
train_data = all_data_reduced[1:seperatingValue, ]
test_data = all_data_reduced[(seperatingValue+1):nrow(all_data_reduced), ]

nrow(train_data)
nrow(test_data)

# 라이브러리 호출
library(h2o)
# h2o framework initialize
h2o.init(nthreads = 2)

# R dataframe type의 데이터를 h2o dataframe 으로 casting
training_frame = as.h2o(train_data)
testing_frame = as.h2o(test_data)

y.dep = 1
x.indep = c(2:length(testing_frame))

# print
y.dep
x.indep

# 모델 생성: tmodel (약 1~2분 소요. 개발 머신의 성능에 따라 다를 수 있음)
tmodel = h2o.deeplearning(x = x.indep, y = y.dep, training_frame = training_frame, hidden = c(32,16), loss = "Automatic", nfolds = 5, epochs = 10)

# testing_frame 내에서 실제 label 값과 모델 예측값과의 차이들을 비교한 성능 평가
tperf = h2o.performance(tmodel, testing_frame)

# 만들어진 모델(tmodel)에 학습에 사용되지 않았던 testing_frame (12,054 dataset)을 label 값을 제외하고 입력으로 넣었을 때 예측된 결과 값: tpredicted
tpredicted = h2o.predict(tmodel, testing_frame)

tmodel
tperf

plot(tmodel,metric='mae')

fepa <- as.data.frame(testing_frame[y.dep])
fepa<- as.data.frame(fepa)
forecast_y_t_hat = tpredicted
forecast_y_t_hat <- as.data.frame(forecast_y_t_hat)


names(fepa)[1] <- c("Actual E/atom ")
names(forecast_y_t_hat) <-c("Predicted E/atom")

vs <- cbind(fepa, forecast_y_t_hat)

plot(vs)
abline(a=0,b=1)

plot(vs, ylim=c(-15, 3), xlim=c(-15,3))
abline(a=0,b=1)

# 전체 data 중 training에 80%, testing에 20%에 사용.
seperatingRate = 0.8
# 이번에는 데이터 전부를 사용함
maxSizeRate = 1

# down scaling
seperatingValue=0
seperatingValue = floor(nrow(all_data)*maxSizeRate*seperatingRate)

set.seed(1234)
all_data_reduced = all_data[sample(nrow(all_data)*maxSizeRate), ]
train_data_full = all_data_reduced[1:seperatingValue, ]
test_data_full = all_data_reduced[(seperatingValue+1):nrow(all_data_reduced), ]

nrow(train_data_full)
nrow(test_data_full)

training_frame_full = as.h2o(train_data_full)
testing_frame_full = as.h2o(test_data_full)

tmodel_full = h2o.deeplearning(x = x.indep, y = y.dep, training_frame = training_frame_full, hidden = c(32,16), loss = "Automatic", nfolds = 5, epochs = 10, variable_importance = TRUE)

tpredicted_full = h2o.predict(tmodel_full, testing_frame_full)
tperf_full = h2o.performance(tmodel_full, testing_frame_full)

tmodel_full
tperf_full

plot(tmodel_full,metric='mae')

fepa_full <- as.data.frame(testing_frame_full[y.dep])
fepa_full<- as.data.frame(fepa_full)
forecast_y_t_hat_full = tpredicted_full
forecast_y_t_hat_full <- as.data.frame(forecast_y_t_hat_full)


names(fepa_full)[1] <- c("Actual E/atom")
names(forecast_y_t_hat_full) <-c("Predicted E/atom")

vs_full <- cbind(fepa_full, forecast_y_t_hat_full)

plot(vs_full, ylim=c(-15, 3), xlim=c(-15,3))
abline(a=0,b=1)

h2o.varimp(tmodel_full)

# save the model to disk
absolute_Path_and_fileName_for_the_Model = h2o.saveModel(tmodel_full, path=paste(currentPath,modelPath, sep=""))

#check file path and name of the saved model 
absolute_Path_and_fileName_for_the_Model

# load the model from disk
loaded_tmodel_full = h2o.loadModel(path=absolute_Path_and_fileName_for_the_Model)

loaded_tpredicted_full = h2o.predict(loaded_tmodel_full, testing_frame_full)
tpredicted_full
loaded_tpredicted_full

fileName = "zero_field_test.csv"
zero_field = read.csv(paste(dataPath, fileName, sep=""),sep=",",header=TRUE)
zero_field

zero_field$Cr <- 2
zero_field$Co <- 1
zero_field$O <- 4

sample_testing_frame = as.h2o(zero_field)
output = h2o.predict(loaded_tmodel_full, sample_testing_frame)
output
