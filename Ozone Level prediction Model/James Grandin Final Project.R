
data <- read.csv("~/Desktop/Machine Learning 1/Final Project/ozone-data.csv")

head(data)
tail(data)
any(is.na(data))
dim(data)
str(data)



###### EDA ########
for (j in 1:ncol(data)){
  if (class(data[[j]]) == "numeric") {
    boxplot(data[[j]], main = colnames(data[j]))
    table <-table(data$Ozone,data[[j]])
    barplot(table,legend.text=rownames(table), main = colnames(data[j]), 
            ylab = "Frequency", width=(length(data[[j]]/20)))
    hist(data[[j]], main = colnames(data[j])) 
    
    #pull only pvalue from the wilcox test save space in the console
    pVal = wilcox.test(data[[j]]~data$Ozone, alternative = 'two.sided')$p.value 
    
    if (pVal<=0.05){A = "Association"}else{A="No Association"}
    print(c(colnames(data[j]),pVal,A))  #print list with variable, pvalue, and association
    
  } else { table <-table(data$Ozone,data[[j]])
    barplot(table,legend.text=rownames(table), main = colnames(data[j]), ylab = "Frequency", las =2)
  
    table2 <-table(data[[j]],data$Ozone,useNA="no")
    pVal2 <- fisher.test(table2, simulate.p.value =TRUE)$p.value
  
    if (pVal2<=0.1){B = "Association"}else{B="No Association"}
    print(c(colnames(data[j]),pVal2,B))
  }
}



# Data partition
set.seed(123)
n <- nrow(data)
id.split <- sample(x=1:3, size = n, replace =TRUE, prob=c(0.50, 0.25, 0.25))
data.train <- data[id.split ==1, ]
data.validate <- data[id.split == 2, ]
data.test <- data[id.split == 3, ]

# training Predictor variables
X.train <- model.matrix(as.formula(factor(Ozone)~.), data= data.train)[, -1]
X.validate <- model.matrix(as.formula(factor(Ozone)~.), data= data.validate)[, -1]
X.test <- model.matrix(as.formula(factor(Ozone)~.), data= data.test)[, -1]
# training Outcome variable
y.train <- data.train$Ozone
y.validate <- data.validate$Ozone
y.test <- data.test$Ozone


data.pooled <- rbind(data.train,data.validate)
X.pooled<- model.matrix(as.formula(factor(Ozone)~.), data= data.pooled)[, -1]
y.pooled<-data.pooled$Ozone
# Will be using lasso regression to filter out uncorrelated variables 
# This makes the model more simple and readable 
#binomial means we are using Logistic Regression which is a much better
#classifier algorithm compared to linear regression



Lambda <- seq(0.001, 0.5, length.out = 500)

L <- length(Lambda)

OUT <- matrix (0, L, 3)

for (i in 1:L){
  #fitting the lasso logistic regression model to the training data
  fit <- glmnet(x=X.train, y=y.train, family ="binomial", alpha =1, #lasso
                lambda=Lambda[i], standardize=T, thresh = 1e-07, maxit=1000)
  #using model to on validation data to determine the best lambda
  pred <- predict(fit, newx=X.validate, s=Lambda[i], type="response")
  miss.rate <- mean(y.validate != (pred > 0.5))
  mse <- mean((y.validate - pred)^2)
  OUT[i, ] <- c(Lambda[i], miss.rate, mse)
  
}



# print the MSE, Missclassification rate table corresponding lambda
head(OUT)
par(mfrow = c(1,2))
plot(OUT[, 1], OUT[,2], type = "b", col = "blue", ylab = "Missclassification rate")
plot(OUT[, 1], OUT[,3], type = "b", col = "red", ylab = "MSE")
lambda.best <- OUT[which.min(OUT[, 3]), 1]
miss.rate_Lambda <- OUT[which.min(OUT[, 3]), 2]


fit.best <- glmnet (x=X.pooled, y=y.pooled, family ="binomial", alpha=1,  #LASSO
                    lambda = lambda.best, standardize = T, thresh = 1e-07, maxit=1000)
names(fit.best)
fit.best$beta # Finding important variables.


pred <- predict(fit.best, newx = X.test, s =lambda.best, type="response")
dim(pred)


library(cvAUC)
yobs <- y.test
AUC <- ci.cvAUC(predictions = pred, labels =yobs, folds=1:NROW(data.test), confidence = 0.95)
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
