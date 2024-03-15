install.packages("Hmisc")
install.packages("poLCA")
install.packages("MoEClust")
install.packages("flexCWM")
install.packages("RSNNS")
install.packages("Metrics")
library(readxl)
library(poLCA)
library(Hmisc)
library(flexCWM)
library(MoEClust)
library(mclust)
library(mixtools)
library(RSNNS)
library(lattice)
library(MASS)

library(Metrics)
[출처] R을 이용해 간단한 신경망 만들기 (12)|작성자 고든
traindata<-read_excel("traindata.xlsx")
testdata<-read_excel("testdata.xlsx")
traindata
RENT_DT<-traindata$대여일자
hist(RENT_DT)
testdata
testdata
View(traindata)
str(traindata)
EXER_AMT<-traindata$EXER_AMT
CARBON_AMT<-traindata$CARBON_AMT
MOVE_TIME<- traindata$MOVE_TIME
GENDER_CD<-traindata$GENDER_CD
RENT_SPOT<-traindata$RENT_SPOT
RENT_HR<-traindata$RENT_HR
USE_CNT<-traindata$USE_CNT
AGE_TYPE<-traindata$AGE_TYPE
RENT_TYPE<-traindata$RENT_TYPE
holiday<-traindata$holiday
park<-traindata$park

EXER_AMT<-traindata$EXER_AMT
CARBON_AMT<-traindata$CARBON_AMT
MOVE_TIME<- traindata$MOVE_TIME
GENDER_CD<-as.factor(traindata$GENDER_CD)
RENT_SPOT<-as.factor(traindata$RENT_SPOT)
RENT_HR<-as.factor(traindata$RENT_HR)
USE_CNT<-as.factor(traindata$USE_CNT)
AGE_TYPE<-as.factor(traindata$AGE_TYPE)
RENT_TYPE<-as.factor(traindata$RENT_TYPE)
holiday<-as.factor(traindata$holiday)
park<-as.factor(traindata$park)
RENT_DT<-as.factor(traindata$대여일자)
traindata111<-data.frame(EXER_AMT=EXER_AMT, AGE_TYPE=AGE_TYPE ,RENT_HR=RENT_HR, RENT_TYPE=RENT_TYPE, park=park ,holiday=holiday ,RENT_SPOT=RENT_SPOT,GENDER_CD=GENDER_CD ,USE_CNT=USE_CNT,CARBON_AMT=CARBON_AMT,MOVE_TIME=MOVE_TIME)
traindata111
summary(traindata111)

write.csv(traindata111,"traindata111.csv")
write.csv(testdata111, "testdata111.csv")

EXER_AMT2<-testdata[1:200,]$EXER_AMT
AGE_TYPE2<-as.factor(testdata[1:200,]$AGE_TYPE)
RENT_HR2<-as.factor(testdata[1:200,]$RENT_HR)
RENT_TYPE2<-as.factor(testdata[1:200,]$RENT_TYPE)
park2<-as.factor(testdata[1:200,]$park)
holiday2<-as.factor(testdata[1:200,]$holiday)
RENT_SPOT2<-as.factor(testdata[1:200,]$RENT_SPOT)
GENDER_CD2<-as.factor(testdata[1:200,]$GENDER_CD)
USE_CNT2<-as.factor(testdata[1:200,]$USE_CNT)
CARBON_AMT2<-testdata[1:200,]$CARBON_AMT
MOVE_TIME2<-testdata[1:200,]$MOVE_TIME

testdata111<-data.frame(EXER_AMT=EXER_AMT2, AGE_TYPE=AGE_TYPE2 ,RENT_HR=RENT_HR2, RENT_TYPE=RENT_TYPE2, park=park2 ,holiday=holiday2 ,RENT_SPOT=RENT_SPOT2,GENDER_CD=GENDER_CD2 ,USE_CNT=USE_CNT2,CARBON_AMT=CARBON_AMT2,MOVE_TIME=MOVE_TIME2)

hist(RENT_DT,main='대여일자')
hist(RENT_HR,main='대여시간')
RENT_DT<-as.numeric(RENT_DT)

regmixmodel.sel(CARBON_AMT+MOVE_TIME,EXER_AMT , w = NULL, k = 4)
regm1<-regmixEM(EXER_AMT, CARBON_AMT+MOVE_TIME, lambda = NULL, beta = NULL, sigma = NULL, k = 3,
         addintercept = TRUE, arbmean = TRUE, arbvar = TRUE,
         epsilon = 1e-08, maxit = 10000, verb = FALSE)
summary(regm1)
plot(CARBON_AMT,regm1$y)
plot(MOVE_TIME,regm1$y)
regm1$x

regm1$posterior
regm2$posterior
regm3$posterior
(regm1$posterior[,max(1,2,3)])

regm1$y
regm2$y
plot(regm1)
traindata
gm1<-Mclust(traindata[,c(18,19)], G = 2)
gm2<-Mclust(traindata[,c(18,19)], G = 3)
gm3<-Mclust(traindata[,c(18,19)], G = 4)
gm1
gm2
gm2
summary(gm1)
summary(gm2)
summary(gm3)

index1<-which(gm1$classification==1)
index2<-which(gm1$classification==2)
group1<-traindata[index1,]
group2<-traindata[index2,]



plot(gm1)

m1<-MoE_clust(EXER_AMT,G=2,gating= ~AGE_TYPE +RENT_HR + RENT_TYPE + park + holiday + RENT_SPOT + GENDER_CD + USE_CNT, expert= ~ CARBON_AMT + MOVE_TIME)
m2<-MoE_clust(EXER_AMT,G=3,gating= ~AGE_TYPE +RENT_HR + RENT_TYPE + park + holiday + RENT_SPOT + GENDER_CD + USE_CNT, expert= ~ CARBON_AMT + MOVE_TIME)
m3<-MoE_clust(EXER_AMT,G=4,gating= ~AGE_TYPE +RENT_HR + RENT_TYPE + park + holiday + RENT_SPOT + GENDER_CD + USE_CNT, expert= ~ CARBON_AMT + MOVE_TIME)
m4<-MoE_clust(EXER_AMT,G=5,gating= ~AGE_TYPE +RENT_HR + RENT_TYPE + park + holiday + RENT_SPOT + GENDER_CD + USE_CNT, expert= ~ CARBON_AMT + MOVE_TIME)

MoE_compare(m2, m3, m4,criterion="bic")
pred1<-predict(m1)
pred2<-predict(m1,newdata=testdata[1:100,])
res<-residuals(m1,testdata[1:200,])
rmse<-sum(res^2)/200
plot(pred1)
plot(m1)
m1
m1$classification
index1<-which(m2$classification==1)
index2<-which(m2$classification==2)
index3<-which(m2$classification==3)
group1<-traindata[index1,]
group2<-traindata[index2,]
group3<-traindata[index3,]
summary(m1)
summary(m2)
m4
m1$BIC



index1
plot(1,type="n",ann=FALSE,
     
     xlim=c(0,10),
     
     ylim=c(0,800))

points(group1$CARBON_AMT,group1$MOVE_TIME,col='blue')
points(group2$CARBON_AMT,group2$MOVE_TIME,col='red')
points(group3$MOVE_TIME,group3$EXER_AMT,col='green')
title(main="",
      
      xlab="CARBON_AMT",
      
      ylab="MOVE_TIME")

legend(8,600,c("cluster1","cluster2"),fill=c("blue","red"))

cloud(group1$EXER_AMT~group1$CARBON_AMT*group1$MOVE_TIME)
cloud(group2$EXER_AMT~group2$CARBON_AMT*group2$MOVE_TIME)

plot(1,type="n",ann=FALSE,
     
     xlim=c(0,800),
     
     ylim=c(0,10))

points(group1$MOVE_TIME,group1$CARBON_AMT,col='blue')
points(group2$MOVE_TIME,group2$CARBON_AMT,col='red')
points(group3$MOVE_TIME,group3$EXER_AMT,col='green')
title(main="",
      
      xlab="CARBON_AMT",
      
      ylab="MOVE_TIME")

legend(8,700,c("group1","group2"),fill=c("blue","red"))




m1$classification     

mean(group1$MOVE_TIME)
mean(group2$MOVE_TIME)
mean(group3$MOVE_TIME)
mean(group1$CARBON_AMT)
mean(group2$CARBON_AMT)
mean(group3$CARBON_AMT)
mean(group1$EXER_AMT)
mean(group2$EXER_AMT)
mean(group3$EXER_AMT)
prop.table(table(group1$AGE_TYPE))
prop.table(table(group2$AGE_TYPE))
prop.table(table(group3$AGE_TYPE))
prop.table(table(group1$GENDER_CD))
prop.table(table(group2$GENDER_CD))
prop.table(table(group3$GENDER_CD))
prop.table(table(group1$RENT_SPOT))
prop.table(table(group2$RENT_SPOT))
prop.table(table(group3$RENT_SPOT))
prop.table(table(group1$RENT_HR))
prop.table(table(group2$RENT_HR))
prop.table(table(group3$RENT_HR))
prop.table(table(group1$RENT_TYPE))
prop.table(table(group2$RENT_TYPE))
prop.table(table(group3$RENT_TYPE))
prop.table(table(group1$holiday))
prop.table(table(group2$holiday))
prop.table(table(group3$holiday))
prop.table(table(group1$park))
prop.table(table(group2$park))
prop.table(table(group3$park))
prop.table(table(group1$USE_CNT))
prop.table(table(group2$USE_CNT))
prop.table(table(group3$USE_CNT))

pred2<-predict(m2,newdata=testdata111)

RMSE(pred2$y,testdata111$EXER_AMT)
MAE(pred2$y, testdata111$EXER_AMT)


cm1<-cwm(formulaY = EXER_AMT ~ AGE_TYPE +RENT_HR + RENT_TYPE + park + holiday + RENT_SPOT + GENDER_CD + USE_CNT + CARBON_AMT + MOVE_TIME, familyY = gaussian, traindata, Xnorm = CARBON_AMT+MOVE_TIME, Xbin = AGE_TYPE + RENT_TYPE +park+holiday+GENDER_CD+USE_CNT,
    Xpois = NULL, Xmult = RENT_HR + RENT_SPOT, modelXnorm = NULL, Xbtrials = NULL, k = 1:3,
    initialization = c("random.soft", "random.hard", "kmeans", "mclust", "manual"),
    start.z = NULL, seed = NULL, maxR = 1, iter.max = 1000, threshold = 1.0e-04,
    eps = 1e-100, parallel = FALSE, pwarning = FALSE)
cm1     
cm1
cm1$formulaY
cm1$call
cm1$models
plot(cm1)
summary(cm1)
cm1$classification


if(!require("caret")){install.packages("caret"); library(caret)}
if(!require("dplyr")){install.packages("dplyr"); library(dplyr)}
if(!require("rpart")){install.packages("rpart"); library(rpart)}
if(!require("rpart.plot")){install.packages("rpart.plot"); library(rpart.plot)}
if(!require("C50")){install.packages("C50"); library(C50)}
if(!require("Epi")){install.packages("Epi"); library(Epi)}
if(!require("ROCR")){install.packages("ROCR"); library(ROCR)}
if(!require("caret")){install.packages("caret"); library(caret)}
if(!require("dplyr")){install.packages("dplyr"); library(dplyr)}
if(!require("party")){install.packages("party"); library(party)}
if(!require("mlbench")){install.packages("mlbench"); library(mlbench)}
if(!require("randomForest")){install.packages("randomForest"); library(randomForest)}



fit_ctree <- ctree(EXER_AMT~ AGE_TYPE +RENT_HR + RENT_TYPE + park + holiday + RENT_SPOT + GENDER_CD + USE_CNT + CARBON_AMT + MOVE_TIME, data=traindata)
fit_ctree
plot(fit_ctree)
fit_ctree$bic
summary(fit_ctree)
pred_fit_ctree<-predict(fit_ctree,testdata[1:200,])
pred_fit_ctree
summary(pred_fit_ctree)
RMSE(pred_fit_ctree, testdata[1:200,]$EXER_AMT)
MAE(pred_fit_ctree, testdata[1:200,]$EXER_AMT)


rf1 <- randomForest(EXER_AMT~ as.factor(AGE_TYPE) +as.factor(RENT_HR) + as.factor(RENT_TYPE) + park + holiday + RENT_SPOT + GENDER_CD + USE_CNT + CARBON_AMT + MOVE_TIME, data=traindata, ntree=200, mtry=2, importance=T)
rf2 <- randomForest(EXER_AMT~ AGE_TYPE +RENT_HR + RENT_TYPE + park + holiday + RENT_SPOT + GENDER_CD + USE_CNT + CARBON_AMT + MOVE_TIME, data=traindata111, ntree=500, mtry=2, importance=T)
pred_rf1<-predict(rf1,testdata111)
pred_rf2<-predict(rf2,testdata111)
RMSE(pred_rf1,testdata[1:200,]$EXER_AMT)
MAE(pred_rf1,testdata[1:200,]$EXER_AMT)
RMSE(pred_rf2,testdata[1:200,]$EXER_AMT)
MAE(pred_rf2,testdata[1:200,]$EXER_AMT)

if(!require("nnet")){install.packages("nnet"); library(nnet)}

if(!require("devtools")){install.packages("devtools"); library(devtools)}
if(!require("reshape2")){install.packages("reshape2"); library(reshape2)}
if(!require("dplyr")){install.packages("dplyr"); library(dplyr)}
if(!require("caret")){install.packages("caret"); library(caret)}
# 신경망 모델에서 각 변수의 중요도 확인
if(!require("NeuralNetTools")){install.packages("NeuralNetTools"); library(NeuralNetTools)}
# 데이터 불러오기



nn <- nnet(EXER_AMT~., data = traindata111, size = 10)
pred_nn<-predict(nn,newdata=testdata111[,-1])
RMSE(pred_nn,testdata111$EXER_AMT)
MAE(pred_nn,testdata111$EXER_AMT)
plot(nn)

mod<-mlp(AGE_TYPE+RENT_HR+RENT_TYPE+park+holiday+RENT_SPOT+GENDER_CD+USE_CNT+CARBON_AMT_MOVE_TIME, EXER_AMT, size=c(10,10,10), maxit = 1000)
plot(mod)
[출처] R을 이용해 간단한 신경망 만들기 (12)|작성자 고든

newtraindata<-read_excel("newtraindata.xlsx")
newtestdata<-read_excel("newtestdata.xlsx")

lm1<-lm(EXER_AMT~AGE_1+AGE_2+AGE_3+AGE_4+AGE_5+RENT_HR1+RENT_HR2+RENT_HR3+RENT_HR4+RENT_HR5+RENT_HR6+RENT_TYPE+park+holiday+RENT_SPOT1+RENT_SPOT2+RENT_SPOT3+GENDER_CD+USE_CNT+CARBON_AMT+MOVE_TIME,newtraindata)
pred_lm1<-predict(lm1,newtestdata)
RMSE(pred_lm1,newtestdata$EXER_AMT)
MAE(pred_lm1,newtestdata$EXER_AMT)
