library(dplyr)
library(forecast)
library(TSA)
library(tseries)
library(imputeTS)
library("xts")
library(quantmod)
library(magrittr) # for the pipe %>% function
library(timetk)
library(padr)

setwd("~/Desktop/Time Series Class/Final Project")
googl_data = read.csv("GOOG.csv")

my_date <- as.Date(googl_data[['Date']],format = "%Y-%m-%d")

#### Plotting time series
googl_xts <- xts(googl_data['Close'],my_date)
googl_xts


par(mfrow = c(3,1))


plot(googl_xts, main = 'Google Stock Price', y_label = "Stock Price $", x_label = "Date")


googl.series <- ts(googl_data$Close, start=c(2006,1,2), end=c(2017, 12, 28), frequency=52)


###Decompose
decomp1 <- decompose(googl.series)
plot(decomp1)
acf(coredata(googl.series), main='ACF - Data1')  
pacf(coredata(googl.series), main='PACF - Data1')

### AR
par(mfrow =c(3,1))
diff_googl <- diff(googl.series, differences = 2)
decomp <- decompose(diff_googl)
plot(decomp)


acf(coredata(diff_googl),type="correlation",plot=T, main='ACF - Data1')  
pacf(coredata(diff_googl), main='PACF - Data1')


### ARIMA (Second Differencing)

googl.arima1 <- arima(googl.series , order = c(1, 2, 1))
googl.arima1

### Best Fit SARIMA


get.best.arima <- function(x.ts, maxord = c(1,1,1,1,1,1)){
  best.aic <- 10^9
  n <- length(x.ts)
  for(p in 0:maxord[1])
    for(d in 0:maxord[2])
      for(q in 0:maxord[3])
        for(P in 0:maxord[4])
          for(D in 0:maxord[5])
            for(Q in 0:maxord[6]){
              fit <- arima(x.ts, order = c(p,d,q), 
                           seas = list(order=c(P,D,Q),
                                       frequency(x.ts)), method="CSS")
              fit.aic <- -2*fit$loglik + (log(n) + 1)* length(fit$coef)
              if(fit.aic < best.aic){
                best.aic <- fit.aic
                best.fit <- fit
                best.model <- c(p,d,q,P,D,Q)
              }
            }
  list(best.aic, best.fit, best.model)
}


googl.best <- get.best.arima(googl.series, maxord=rep(2,6)) 
googl.best
googl.arima2 <- arima(googl.series, order=c(0,2,0), seas=list(order=c(1,2,1), 52))
googl.arima2
t(confint(googl.arima2))

## determining if arch/garch will be useful
checkresiduals(googl.arima2)
par(mfrow=c(3,1))
ts.plot(googl.arima2$residuals)
acf(googl.arima2$residuals)
acf(googl.arima2$residuals^2)


### Arch/Garch
par(mfrow=c(3,1))
googl.garch<-garch(order = c(1,22), googl.arima2$residuals, trace=F)
googl.garch
t(confint(googl.garch))

googl.garch.res <- resid(googl.garch)[-1]
acf(googl.garch.res)
acf(googl.garch.res^2)
#### Try BoxCox Transform Transform

googl_boxcox['Close'] = boxcox(as.vector(googl_data['Close']))


class(googl_data['Close'])
library(forecast)

lambda <- BoxCox.lambda(googl.series)
plot.ts(BoxCox(googl.series, lambda = lambda))
googl.bestBoxcox <- get.best.arima(BoxCox(googl.series, lambda = lambda), maxord=rep(2,6)) 
googl.bestBoxcox
googl.arima3 <- arima(BoxCox(googl.series, lambda = lambda), order=c(0,1,0))
googl.arima3
t(confint(googl.arima3))

checkresiduals(googl.arima3)
par(mfrow=c(3,1))
ts.plot(googl.arima3$residuals)
acf(coredata(googl.arima3$residuals))
acf(coredata(googl.arima3$residuals^2))



## Checking if residuals From boxcox transform model is stationary

adf.test(googl.arima3$residuals)

forecast_data <- forecast(googl.arima3) 
print(forecast_data)
plot(forecast_data, main = "forecasting_data for rain_ts") 
