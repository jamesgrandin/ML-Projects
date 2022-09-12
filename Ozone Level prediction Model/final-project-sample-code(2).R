################## Bring the Data set into R ###################
install.packages("mlbench")
data(BreastCancer, package="mlbench")
data <-BreastCancer

# removing column id as it is not important for data analysis
dat <- data[, -1]
dim(dat)

# make the class variable into binary
dat$y <- ifelse(dat$Class=="benign", 1, 0)
dim(dat)

# delete the class cokumn as we have a new y column.
dat<- dat[, -10]

# check the observations of entire dataset
for (j in 1:NCOL(dat)){
  print(colnames(dat)[j])
  print(table(dat[,j], useNA="ifany"))
}

# make the character value into numeric
dat <- apply(dat, 2, FUN=function(x) {as.numeric(as.character(x))})
dat <-na.omit(dat)
dat <- as.data.frame(dat)


# Data partition
set.seed(123)
n <- nrow(dat)
id.split <- sample(x=1:2, size = n, replace =TRUE, prob=c(0.75, 0.25))
dat.train <- dat[id.split ==1, ]
dat.test <- dat[id.split == 2, ]



# training Predictor variables
X.train <- model.matrix(as.formula(factor(y)~.), data= dat.train)[, -1]
# training Outcome variable
y.train <- dat.train$y


# Fit the logistic regression with different lambda
Lambda <- seq(0.0001, 0.5, length.out = 200)
L <- length(Lambda)
OUT <- matrix (0, L, 3)
for (i in 1:L){
  fit <- glmnet(x=X.train, y=y.train, family ="binomial", alpha =1, #lasso
                lambda=Lambda[i], standardize=T, thresh = 1e-07, maxit=1000)
  pred <- predict(fit, newx=X.train, s=Lambda[i], type="response")
  miss.rate <- mean(y.train != (pred > 0.5))
  mse <- mean((y.train - pred)^2)
  OUT[i, ] <- c(Lambda[i], miss.rate, mse)
  
}

# print the MSE, Missclassification rate table corresponding lambda
head(OUT)
par(mfrow = c(1,2))
plot(OUT[, 1], OUT[,2], type = "b", col = "blue", ylab = "Missclassification rate")
plot(OUT[, 1], OUT[,3], type = "b", col = "red", ylab = "MSE")
(lambda.best <- OUT[which.min(OUT[, 3]), 1])
(miss.rate_Lambda <- OUT[which.min(OUT[, 3]), 2])

####### Fit the Logistic regression with best lambda
fit.best <- glmnet (x=X.train, y=y.train, family ="binomial", alpha=1,  #LASSO
                    lambda = lambda.best, standardize = T, thresh = 1e-07, maxit=1000)
names(fit.best)
fit.best$beta # Finding important variables.


# Test input feature
X.test <- model.matrix(as.formula(factor(y)~.), data= dat.test)[, -1]
# test Outcome variable
y.test <- dat.test$y

# Prediction
pred <- predict(fit.best, newx = X.test, s =lambda.best, type="response")
dim(pred)

#########################################################################
######################### Model Evaluation ##############################
########################################################################
library(cvAUC)
yobs <- dat.test$y
AUC <- ci.cvAUC(predictions = pred, labels =yobs, folds=1:NROW(dat.test), confidence = 0.95)
AUC
auc.ci <- round(AUC$ci, digits = 3)

library(verification)
mod.glm <- verify(obs = yobs, pred = pred)
roc.plot(mod.glm, plot.thres=NULL)
text(x=0.7, y=0.2, paste("Area under ROC = ", round(AUC$cvAUC, digits = 3), "with 95% CI (",
                         auc.ci[1], ",", auc.ci[2], ").", sep = " "), col="blue", cex =1.2)

library(caret)
pred1 <- ifelse(pred>0.5, 1, 0)
confusionMatrix(factor(pred1), factor(y.test))








