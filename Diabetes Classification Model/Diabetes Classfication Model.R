data <- read.csv("diabetes.csv")
head(data)


# Dependencies
require(dplyr)
library(mice, quietly=TRUE)
require(caret)
library(cvAUC)
library(verification)
library(e1071)
library(generalhoslem)
library(ResourceSelection)
library(corpcor)
library(naniar)
library(glmnet)
library(ggplot2)
library(cowplot) 


vis_miss(data)
gg_miss_var(data)


## This is where I would Impute NA values if there were any


#fit.mice <- mice(data, m=1, maxit=50, method='pmm', seed=5474, printFlag=FALSE) 
#data <- complete(fit.mice, 1)
#confirm imputations have been completed on NA data
#vis_miss(data)
#gg_miss_var(data)


#### EDA ######


## This is where I would Impute NA values 

data$Glucose[data$Glucose == 0] <- NA
data$BloodPressure[data$BloodPressure == 0] <- NA
data$SkinThickness[data$SkinThickness == 0] <- NA
data$Insulin[data$Insulin == 0] <- NA
data$BMI[data$BMI == 0] <- NA


fit.mice <- mice(data, m=1, maxit=50, method='pmm', seed=5474, printFlag=FALSE) 
data <- complete(fit.mice, 1)

#confirm imputations have been completed on NA data
#vis_miss(data)
#gg_miss_var(data)


#### EDA ######

par(mfrow = c(3,3))

my_plots <- lapply(names(data), function(var_x){
  p <- 
    ggplot(data) +
    aes_string(var_x)
  
  if(is.numeric(data[[var_x]])) {
    p <- p + geom_histogram()
  } else {
    p <- p + geom_bar() #+ scale_y_continuous(limits = c(0 , 50))
  } 
  
})

plot_grid(plotlist = my_plots)




######################### Data partition ##################



index <- createDataPartition(data$Outcome, p = .70, list = FALSE)
train <- data[index, ]
test <- data[-index, ]
test <-subset(test, select = -12)
y.train <- train$Outcome
yobs <-test$Outcome




par(mfrow = c(2,3))
##################### MODEL 1 ############################
############################################################

logitmodel <- glm(as.factor(Outcome) ~ ., family = binomial(link='logit'), train)
summary(logitmodel)

pred <- predict(logitmodel,test, type = "response")

pred1 <- ifelse(pred>0.5, 1, 0)
table(pred1)

## Missclasification rate
(miss.rate <- mean(yobs != pred1))


#Plotting ROC curve of the fit.best model.
AUC <- ci.cvAUC(predictions = pred, labels = yobs, folds=1:NROW(test), confidence = 0.95)
AUC
(auc.ci <- round(AUC$ci, digits = 3))

logit.glm <- verify(obs = yobs, pred = pred)
roc.plot(logit.glm, plot.thres=NULL)
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white")
title("ROC Model 1")
text(x=0.7, y=0.2, paste("AUC = ", round(AUC$cvAUC, digits = 3),sep = " "), col="blue", cex =1.2)
text(x=0.7, y = 0.1, paste("Miss Rate = ",round(miss.rate, digits = 4),sep = " "))





##################### MODEL 2 Drop Insignificant Coefficients ############################
############################################################

logitmodel <- glm(as.factor(Outcome) ~ Pregnancies + Glucose + BMI + DiabetesPedigreeFunction, family = binomial(link='logit'), train)
summary(logitmodel)

pred <- predict(logitmodel,test, type = "response")

pred1 <- ifelse(pred>0.5, 1, 0)
table(pred1)

## Missclasification rate
(miss.rate <- mean(yobs != pred1))


#Plotting ROC curve of the fit.best model.
AUC <- ci.cvAUC(predictions = pred, labels = yobs, folds=1:NROW(test), confidence = 0.95)
AUC
(auc.ci <- round(AUC$ci, digits = 3))

logit.glm <- verify(obs = yobs, pred = pred)
roc.plot(logit.glm, plot.thres=NULL)
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white")
title("ROC Model 2 (Dropped Coefficients")
text(x=0.7, y=0.2, paste("AUC = ", round(AUC$cvAUC, digits = 3),sep = " "), col="blue", cex =1.2)
text(x=0.7, y = 0.1, paste("Miss Rate = ",round(miss.rate, digits = 4),sep = " "))



######### MODEL 3  Log transform right skewed data ################
###################################################################

logdata <- data

logdata$Insulin <-log(logdata$Insulin)
logdata$DiabetesPedigreeFunction <-log(logdata$DiabetesPedigreeFunction)


index <- createDataPartition(logdata$Outcome, p = .70, list = FALSE)
train <- logdata[index, ]
test <- logdata[-index, ]
test <-subset(test, select = -12)
y.train <- train$Outcome
yobs <-test$Outcome



logitmodellog <- glm(as.factor(Outcome) ~ ., family = binomial(link='logit'), train)
summary(logitmodellog)


pred <- predict(logitmodellog,test, type = "response")

pred1 <- ifelse(pred>0.5, 1, 0)
table(pred1)

## Missclasification rate
(miss.rate <- mean(yobs != pred1))


#Plotting ROC curve of the fit.best model.
AUC <- ci.cvAUC(predictions = pred, labels = yobs, folds=1:NROW(test), confidence = 0.95)
AUC
(auc.ci <- round(AUC$ci, digits = 3))

logit.glm <- verify(obs = yobs, pred = pred)
roc.plot(logit.glm, plot.thres=NULL)
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white")
title("ROC Model 3 Log Transform")
text(x=0.7, y=0.2, paste("AUC = ", round(AUC$cvAUC, digits = 3),sep = " "), col="blue", cex =1.2)
text(x=0.7, y = 0.1, paste("Miss Rate = ",round(miss.rate, digits = 4),sep = " "))

