library(ggplot2)
library(reshape)
library(corrplot)
library(forecast)

filename <- "../data/data_waterlevel_final_clear"

#format date and datetime
data_air <- read.csv(filename, header = TRUE)
data_air$Tanggal <- as.Date(as.character(data_air$Tanggal), "%d/%m/%Y")

data_air$datetime <- paste(data_air$Tanggal,data_air$Jam, sep = " ")
data_air$datetime <- as.POSIXct(data_air$datetime, format = "%Y-%m-%d %H:%M")

data_ts <- cbind(data_air[,15], (data_air[,4:14]))
names(data_ts)[1] <- "Datetime"

#summary data
summary(data_ts)
data_ts2 <- data_ts[, -c(2,3,4, 5,9, 12)]
data_pos1 <- melt(data_ts2 ,  id.vars = "Datetime")
names(data_pos1) <- c("Datetime", "Pintu_Air", "Ketinggian_Air")
ggplot(data_pos1, aes(Datetime,Ketinggian_Air)) + 
        geom_line(aes(colour = Pintu_Air))

plot(data_air$Pos.Depok ~ data_air$datetime, type = "l")

#plot correlation
M <- cor(data_ts[,-1])
corrplot.mixed(M)

#cek 1 pintu air
par(mar=c(5,5,5,5), mfrow= c(1,1))
plot(data_ts$Pos.Krukut.Hulu ~ data_ts$Datetime, type = "l")
abline(reg=lm(data_ts$Pos.Krukut.Hulu ~ data_ts$Datetime))

plot(diff(data_ts$Bendung.Katulampa), type = "l")
abline(reg=lm(data_ts$Pos.Krukut.Hulu ~ data_ts$Datetime))

#plot(stl(diff(data_ts$Pos.Krukut.Hulu),s.window="periodic"))

time_index <- seq(from = data_ts$Datetime[1], 
                  to = data_ts$Datetime[2137], by = "hour")

ts <- as.xts(data_ts$Pos.Krukut.Hulu, order.by = time_index)
summary(ts)
cycle(ts)
boxplot(ts~cycle(ts))
frequency(ts)
#par(mar=c(5,5,5,5), mfrow= c(1,1))
#ACF & PACF
par(mar=c(3,3,3,3), mfrow= c(3,2))
for (name in names(data_ts[2:4]))
{
  acf(diff(data_ts[,name]),lag.max = 10, main=name)
  pacf(diff(data_ts[,name]),lag.max = 10, main=name)
}
for (name in names(data_ts[5:7]))
{
  acf(data_ts[,name],lag.max = 10, main=name)
  pacf(data_ts[,name],lag.max = 10, main=name)
}
par(mar=c(3,3,3,3), mfrow= c(2,1))
#arima
fit <- arima(data_ts$Bendung.Katulampa[1:2120], order = c(1,0,0))
#tsdiag(fit)
arimaorder(fit) #p d  q
accuracy(fit)
ts_pred <- predict(fit, n.ahead = 20)
plot(data_ts$Bendung.Katulampa, type = 'p',xlim=c(2110,2145),ylim=c(0,60), 
     main = "ARIMA(1,0,0)")
lines(ts_pred$pred,col="red")
lines(ts_pred$pred+2*ts_pred$se,col="red",lty=3)
lines(ts_pred$pred-2*ts_pred$se,col="red",lty=3)

#autoARMA
fit.a <- auto.arima(data_ts$Bendung.Katulampa[1:2120])
#tsdiag(fit.a)
arimaorder(fit.a) #p d  q
accuracy(fit.a)
ts_pred_a <- predict(fit.a, n.ahead = 20)
plot(data_ts$Bendung.Katulampa, type = 'p',xlim=c(2110,2145),ylim=c(0,60),
     main = "ARIMA(1,1,2)")
lines(ts_pred_a$pred,col="red")
lines(ts_pred_a$pred+2*ts_pred$se,col="red",lty=3)
lines(ts_pred_a$pred-2*ts_pred$se,col="red",lty=3)


#arima
fit <- arima(data_ts$Pos.Krukut.Hulu[1:2100], order = c(4,0,0))
#tsdiag(fit)
arimaorder(fit) #p d  q
accuracy(fit)
ts_pred <- predict(fit, n.ahead = 37)
plot(data_ts$Pos.Krukut.Hulu~data_ts$Datetime, type = 'p',
     xlim=c(data_ts$Datetime[2100],data_ts$Datetime[2137]),
     ylim=c(30,180),
     main = "ARIMA(4,0,0)")
lines(ts_pred$pred~data_ts$Datetime[2101:2137],col="red")
lines(ts_pred$pred+2*ts_pred$se~data_ts$Datetime[2101:2137],col="red",lty=3)
lines(ts_pred$pred-2*ts_pred$se~data_ts$Datetime[2101:2137],col="red",lty=3)

#autoARMA
fit.a <- auto.arima(data_ts$Pos.Krukut.Hulu[1:2100])
#tsdiag(fit.a)
arimaorder(fit.a) #p d  q
accuracy(fit.a)
ts_pred_a <- predict(fit.a, n.ahead = 37)
plot(data_ts$Pos.Krukut.Hulu~data_ts$Datetime, type = 'p',
     xlim=c(data_ts$Datetime[2100],data_ts$Datetime[2137]),
     ylim=c(30,180),
     main = "ARIMA(0,1,3)")
lines(ts_pred_a$pred~data_ts$Datetime[2101:2137],col="red")
lines(ts_pred_a$pred+2*ts_pred$se~data_ts$Datetime[2101:2137],col="red",lty=3)
lines(ts_pred_a$pred-2*ts_pred$se~data_ts$Datetime[2101:2137],col="red",lty=3)
