
# path setting
currentPath = "./"
dataPath = "data/"
figPath = ".figures/"
modelPath = "models/"

dataPath <- paste(currentPath, dataPath, sep="")
figPath <- paste(currentPath, figPath, sep="")
modelPath <- paste(currentPath, modelPath, sep="")

fileName = "reference_elements_dataset.csv"
atomtable = read.csv(paste(dataPath,fileName, sep=""), sep=",", header=TRUE)
#데이터 확인
atomtable

# 전체 data 중 training에 80%, testing에 20%에 사용.
seperatingRate = 0.8
maxSizeRate = 1
all_data = 0
fileName = "oqmd-complex-removedoutliers.csv"
all_data = read.csv(paste(dataPath,fileName,sep=""), sep=",", header=TRUE)

# 데이터 확인
all_data
attach(all_data)

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

# categorical data 처리
all_data["spacegroupnum"] <- as.factor(all_data$spacegroupnum)

# down scaling
seperatingValue = 0
seperatingValue = floor(nrow(all_data)*maxSizeRate*seperatingRate)

# train/test data 분리
set.seed(1234)
all_data_reduced = all_data[sample(nrow(all_data)*maxSizeRate), ]
train_data = all_data_reduced[1:seperatingValue, ]
test_data = all_data_reduced[(seperatingValue+1):nrow(all_data_reduced), ]

# 라이브러리 호출
library(h2o)
# h2o framework initialize
h2o.init(nthreads = 2)

# R dataframe type의 데이터를 h2o dataframe 으로 casting
training_frame = as.h2o(train_data)
testing_frame = as.h2o(test_data)

y.dep=3
x.indep=c(5:length(testing_frame))

# print
y.dep
x.indep

# 모델 생성: tmodel (약 33분 소요. 개발 머신의 성능에 따라 다를 수 있음)
# proc.time()을 이용하여 모델 생성 시간 측정
ptm <- proc.time()
tmodel=h2o.deeplearning(x=x.indep, y=y.dep, training_frame=training_frame, hidden=c(512,256,128,64), loss="Automatic", nfolds = 5, epochs = 10)
tpredicted = h2o.predict(tmodel, testing_frame)
tperf = h2o.performance(tmodel, testing_frame)
proc.time() - ptm

tmodel
tperf

plot(tmodel, metric='mae')

fepa <- as.data.frame(testing_frame[y.dep])
fepa<- as.data.frame(fepa)
forecast_y_t_hat = tpredicted
forecast_y_t_hat <- as.data.frame(forecast_y_t_hat)


names(fepa)[1] <- c("Actual E/atom ")
names(forecast_y_t_hat) <-c("Predicted")

vs <- cbind(fepa, forecast_y_t_hat)

plot(vs)
abline(a=0,b=1)
