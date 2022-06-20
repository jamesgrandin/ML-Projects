## Dependencies
library(naniar)
library(mice, quietly=TRUE)
require(dplyr)
require(caret)
library(cvAUC)
library(verification)
library(e1071)
library(generalhoslem)
library(ResourceSelection)
library(corpcor)
library(glmnet)
library(randomForest)

################ Data bringing ##################
setwd('~/Desktop/STAT5290/Final Project')
data <-read.table(file = 'pop_failures.dat', header = TRUE)
dim(data)
head(data)

data<-data[,-1:-2]
head(data)


hist(data$outcome)

vis_miss(data)
gg_miss_var(data)



## Imputing NA values
###fit.mice <- mice(data, m=1, maxit=50, method='pmm', seed=5474, printFlag=FALSE) 
###data <- complete(fit.mice, 1)
#confirm imputations have been completed on NA data
###vis_miss(data)
###gg_miss_var(data)
###head(data$type)

par(mfrow = c(4,5))


for (j in 1:ncol(data)) {
  hist(data[,j] , main = colnames(data)[j])
}

cor(data)



######################### Data partition ##################



set.seed(123)
n <- nrow(data)
split_data <- sample(x=1:2, size = n, replace=TRUE, prob=c(0.67, 0.33))
train <- data[split_data == 1, ]
train <- as.data.frame(lapply(train,as.numeric))
test <- data[split_data == 2, ]
test <- as.data.frame(lapply(test,as.numeric))
y.train <- train$outcome
yobs <- test$outcome



##### Lasso Logistic Regression ################
#####################################################

formula0 <- factor(outcome) ~ .
X <- model.matrix (as.formula(formula0), data = train)


fit.lasso <- glmnet(x=X, y = y.train, family="binomial", alpha=1, 
                    lambda.min = 1e-4, nlambda = 200, standardize=T, thresh = 1e-07, 
                    maxit=1000)

#Using cross validation to determine the optimal tuning parameter.

CV <- cv.glmnet(x=X, y=y.train, family="binomial", alpha = 1, 
                lambda.min = 1e-4, nlambda = 200, standardize = T, thresh = 1e-07, 
                maxit=1000)

coef(CV, CV$lambda.min)
coef(CV, CV$lambda.1se)

b.lambda <- CV$lambda.1se; b.lambda  

pooledTestTrain <- rbind(train,test)
Y <- pooledTestTrain$outcome
X <- model.matrix (as.formula(formula0), data =pooledTestTrain)

fit.best <- glmnet(x=X, y=Y, family="binomial", alpha = 1, 
                   lambda=b.lambda, standardize = T, thresh = 1e-07, 
                   maxit=1000)
(fit.best$beta)

X.test <- model.matrix (as.formula(formula0), data = test)
pred <- predict(fit.best, newx = X.test, s=b.lambda, type="response")



pred1 <- ifelse(pred>0.5, 1, 0)

(miss.rate <- mean(yobs != pred1))
AUC <- ci.cvAUC(predictions = pred, labels = yobs, folds=1:NROW(test), confidence = 0.95)
AUC
(auc.ci <- round(AUC$ci, digits = 3))
par(mfrow = c(3,3))

mod.glm <- verify(obs = yobs, pred = pred)
roc.plot(mod.glm, plot.thres=NULL)
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white")
title("ROC Lasso Logit Regression")
text(x=0.7, y=0.2, paste("AUC = ", round(AUC$cvAUC, digits = 3),sep = " "), col="blue", cex =1.2)
text(x=0.7, y = 0.1, paste("Miss Rate = ",round(miss.rate, digits = 4),sep = " "))



#### Random Forest #################################
##########################################################

set.seed(123)
rf_fit <- randomForest(as.factor(outcome) ~ .,
                       data=train, 
                       importance=TRUE, 
                       proximity=TRUE)

#importance(rf_fit)
#varImpPlot(rf_fit)


rf_yhat <- predict(rf_fit, newdata=test, type="prob")[, 2]


rf_yhat1 <- ifelse(rf_yhat>0.5, 1, 0)
(miss.rate <- mean(yobs != rf_yhat1))

rf_AUC <- ci.cvAUC(predictions = rf_yhat, labels =yobs, folds=1:NROW(test), confidence = 0.95)
rf_AUC
(rf_auc.ci <- round(rf_AUC$ci, digits = 3))


mod.rf <- verify(obs = yobs, pred = rf_yhat)

roc.plot(mod.rf, plot.thres=NULL)
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white")
title("ROC Random Forest")
text(x=0.7, y=0.2, paste("AUC = ", round(rf_AUC$cvAUC, digits = 3),sep = " "), col="blue", cex =1.2)
text(x=0.7, y = 0.1, paste("Miss Rate = ",round(miss.rate, digits = 4),sep = " "))






##### Neural Network ##################################################
########################################################################
#######################################################################

####Neural Net 1 ####################
######################################

library(neuralnet)
#https://www.rdocumentation.org/packages/neuralnet/versions/1.44.2/topics/neuralnet
options(digits=3)
net1 <- neuralnet(outcome ~ ., 
                  data=train, 
                  hidden=3, #1 hidden layer, 3 units
                  act.fct='logistic', err.fct="sse", linear.output=F, likelihood=TRUE)


###Neural net 2##############################
########################################

net2 <- neuralnet(outcome ~ ., 
                  data=train, 
                  hidden=c(10), #1 hidden layer, 3 units
                  act.fct='logistic', err.fct="sse", linear.output=F, likelihood=TRUE)

###Neural Net 3###########################
##########################################

net3 <- neuralnet(outcome ~ ., 
                  data=train, 
                  hidden=c(3,3,3), #1 hidden layer, 3 units
                  act.fct='logistic', err.fct="sse", linear.output=F, likelihood=TRUE)









## Model 1 eval


ypred <- compute(net1, covariate=test)$net.result
ypred <- ifelse(ypred>0.5,1,0)

(miss.rate <- mean(yobs != ypred))


rf_AUC <- ci.cvAUC(predictions = ypred, labels =yobs, folds=1:NROW(test), confidence = 0.95)
rf_AUC
(rf_auc.ci <- round(rf_AUC$ci, digits = 3))


mod.nn <- verify(obs = yobs, pred = ypred)

roc.plot(mod.nn, plot.thres=NULL)
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white")
title("ROC NNet 1")
text(x=0.7, y=0.2, paste("AUC = ", round(rf_AUC$cvAUC, digits = 3),sep = " "), col="blue", cex =1.2)
text(x=0.7, y = 0.1, paste("Miss Rate = ",round(miss.rate, digits = 4),sep = " "))


## Model 2 Eval
ypred <- compute(net2, covariate=test)$net.result
ypred <- ifelse(ypred>0.5,1,0)

(miss.rate <- mean(yobs != ypred))

rf_AUC <- ci.cvAUC(predictions = ypred, labels =yobs, folds=1:NROW(test), confidence = 0.95)
rf_AUC
(rf_auc.ci <- round(rf_AUC$ci, digits = 3))


mod.nn <- verify(obs = yobs, pred = ypred)
roc.plot(mod.nn, plot.thres=NULL)
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white")
title("ROC NNet 2")
text(x=0.7, y=0.2, paste("AUC = ", round(rf_AUC$cvAUC, digits = 3),sep = " "), col="blue", cex =1.2)
text(x=0.7, y = 0.1, paste("Miss Rate = ",round(miss.rate, digits = 4),sep = " "))



## Model 3 Eval
ypred <- compute(net3, covariate=test)$net.result
ypred <- ifelse(ypred>0.5,1,0)


(miss.rate <- mean(yobs != ypred))


rf_AUC <- ci.cvAUC(predictions = ypred, labels =yobs, folds=1:NROW(test), confidence = 0.95)
rf_AUC
(rf_auc.ci <- round(rf_AUC$ci, digits = 3))


mod.nn <- verify(obs = yobs, pred = ypred)
roc.plot(mod.nn, plot.thres=NULL)
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white")
title("ROC NNet 3")
text(x=0.7, y=0.2, paste("AUC = ", round(rf_AUC$cvAUC, digits = 3),sep = " "), col="blue", cex =1.2)
text(x=0.7, y = 0.1, paste("Miss Rate = ",round(miss.rate, digits = 4),sep = " "))

