library(forecast)
library(xts)
library(tseries)

filename <- "../data/data_waterlevel_final_clear"
data.air <- read.csv(filename)
sum(is.na(data.air))
head(data.air)
data.air$Date <- as.Date(data.air$Date, format = "%d/%m/%Y")
data.air$DateTime = paste(data.air$Date, data.air$Time, sep = " ")

for (i in 3:14)
{
  series = as.data.frame(cbind(data.air[,15], data.air[,i]))
  series$V1 <- as.POSIXct(series$V1)
  series$V2 <- as.numeric(series$V2)
  ts.bk <- xts(series$V2, series$V1)
  testStationary <- adf.test(ts.bk, alternative = "stationary")
  testStationary2 <- Box.test(series$V2)
  testStationary3 <- kpss.test(series$V2)
  print(i)
  print(testStationary$p.value)
  print(testStationary2$p.value)
  print(testStationary3$p.value)
}

auto.arima(ts.bk)
plot.xts(ts.bk)

plot.forecast(ts.bk)
acf(ts.bk)

a <- ts(series)

auto.arima(series$V2)
M <- cor(data.air[,-c(1,2, 10)])
corrplot.mixed(M)

