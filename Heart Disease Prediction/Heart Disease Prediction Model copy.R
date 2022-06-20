data <- read.csv('heart.csv')

library(naniar)
vis_miss(data)
gg_miss_var(data)


# Dependencies
require(dplyr)
library(mice, quietly=TRUE)
require(caret)
library(cvAUC)
library(verification)
library(e1071)
library(generalhoslem)
library(ResourceSelection)
########## DATA Cleaning



##Imputiing NA values
#fit.mice <- mice(data, m=1, maxit=50, method='pmm', seed=5474, printFlag=FALSE) 
#data <- complete(fit.mice, 1)



############## EDA ###################################

hist(data$Age)
hist(data$Cholesterol)
hist(data$HeartDisease)
hist(data$RestingBP)
hist(data$FastingBS)

datanumeric <- subset(data, select =c(-2,-3,-5,-6,-7,-9,-11,-12))


## check multicolinearty
cor(datanumeric)

# Splitting the data into train and test
index <- createDataPartition(data$HeartDisease, p = .70, list = FALSE)
train <- data[index, ]
test <- data[-index, ]
y.train <- train$HeartDisease
yobs <-test$HeartDisease
test <-subset(test, select = -12)

## glm function has not tuning parameter so we will leave it as is
## glm familiy set to binonial since the response distribution is binomial

logitmodel <- glm(as.factor(HeartDisease) ~ ., family = binomial(link='logit'), train)
# Checking the model
summary(logitmodel)



pred <- predict(logitmodel,test, type = "response")

pred1 <- ifelse(pred>0.5, 1, 0)
table(pred1)

## Missclasification rate
(miss.rate <- mean(yobs != pred1))


##### Finidng MSE
MSE.a <- mean((yobs-pred)^2)
MSE.a


#Plotting ROC curve of the fit.best model.
AUC <- ci.cvAUC(predictions = pred, labels = yobs, folds=1:NROW(test), confidence = 0.95)
AUC
(auc.ci <- round(AUC$ci, digits = 3))

logit.glm <- verify(obs = yobs, pred = pred)
roc.plot(logit.glm, plot.thres=NULL)
text(x=0.7, y=0.2, paste("AUC = ", round(AUC$cvAUC, digits = 3), "with 95% CI (",
                         auc.ci[1], ",", auc.ci[2], ").", sep = " "), col="blue", cex =1.2)



ytest <-as.factor(yobs)
predtest <- as.numeric(pred1)
predtest <- as.factor(pred1)
#confusion matrix
confusionMatrix(as.factor(yobs), as.factor(pred1))



###### STATS TEST GLOBAL NULL, NULL DEVIANCE<< ETC

### Global Null ###

C=logitmodel$null.deviance - logitmodel$deviance
pchisq(C,df=399-394,lower.tail = F) # Small p-value. We reject the null hypothesis.



#### Goodness of fit

#Hoslem Test

h <- hoslem.test(logitmodel$y, fitted(logitmodel), g=3)
h #Very high p-value. Model fits the data well



######################################################################################
######################################################################################




### Drop insignificant coefficients and compare (Resting BP, RestingECG, MaxHR)
logitmodelrefined <- glm(as.factor(HeartDisease) ~ Age + Sex + ChestPainType 
                         + Cholesterol + FastingBS
                         + ExerciseAngina + ST_Slope, family = binomial(link='logit'), train)

summary(logitmodelrefined)


pred <- predict(logitmodelrefined,test, type = "response")

pred1 <- ifelse(pred>0.5, 1, 0)
table(pred1)

## Missclasification rate
(miss.rate <- mean(yobs != pred1))


##### Finidng MSE
MSE.a <- mean((yobs-pred)^2)
MSE.a


#Plotting ROC curve of the fit.best model.
AUC <- ci.cvAUC(predictions = pred, labels = yobs, folds=1:NROW(test), confidence = 0.95)
AUC
(auc.ci <- round(AUC$ci, digits = 3))

logit.glm <- verify(obs = yobs, pred = pred)
roc.plot(logit.glm, plot.thres=NULL)
text(x=0.7, y=0.2, paste("AUC = ", round(AUC$cvAUC, digits = 3), "with 95% CI (",
                         auc.ci[1], ",", auc.ci[2], ").", sep = " "), col="blue", cex =1.2)



ytest <-as.factor(yobs)
predtest <- as.numeric(pred1)
predtest <- as.factor(pred1)
#confusion matrix
confusionMatrix(as.factor(yobs), as.factor(pred1))


###### STATS TEST GLOBAL NULL, NULL DEVIANCE<< ETC

### Global Null ###

C=logitmodelrefined$null.deviance - logitmodelrefined$deviance
pchisq(C,df=399-394,lower.tail = F) # Small p-value. We reject the null hypothesis.



#### Goodness of fit

#Hoslem Test

h <- hoslem.test(logitmodelrefined$y, fitted(logitmodelrefined), g=3)
h #Very high p-value. Model fits the data well

anova(logitmodel,logitmodelrefined)



