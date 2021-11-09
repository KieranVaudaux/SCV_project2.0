setwd("C:/Users/blert/Desktop/project2/SCV_project2.0/data/observatoire-geneve")
df_temp <- read.delim("TG_STAID000241.txt", header = FALSE, sep=",", skip=20)
colnames(df_temp) <- c("SOUID", "date", "TG", "Q_TG")
df_temp <- df_temp[df_temp$Q_TG != '9' ,]
df_temp$TG<-df_temp$TG/10

df_temp<- subset(df_temp, select=-c(SOUID,Q_TG))
df_temp$date <- as.Date(as.character(df_temp$date), "%Y%m%d")
Mean_temperature <- ts(df_temp$TG, start=c(1901,1,1),frequency=365.25)

df_year<-df_temp
df_year$date<-format(as.Date(df_year$date, format="%Y/%d/%d"),"%Y")

#vector_year<-data.frame(Date=as.Date(character()),maxtemp=double()) 

vector_prov<-vector()
for (i in 1901:2021){
vector_prov[i-1901]<-max(subset(df_year, date==i)$TG)
}
vector_year<-data.frame(maxtemp=vector_prov)
#vector_year$maxtemp<-vector_prov

year_ts<-ts(vector_prov, start=1901, frequency=1)


library(TSstudio)
library(tidyverse)
library(lubridate)
library(timetk)
library(evd)
library(ggfortify)
library(extRemes)

ts_info(Mean_temperature)
ts_plot(Mean_temperature)

ts_info(year_ts)
ts_plot(year_ts)
autoplot(year_ts)

#Extreme Value Analysis of mean temperature from 1901 to 2020
#Dependence on time, seasonal effect, tendency in climate
#Statistical procedures to remove a trend or a seasonal component.
#Autocorrelation function

#Extremes of dependent sequences
#Event X_i>u and X_j>u are approximateley independent u is high enough
#Dependence in stationary series impose some constraints
#knowledge that the temperature is extreme will influence the probability that it will 
#be extreme in 1 or 2 days, but not in three months

#The D(un) condition for stationary series X_1, X_2, ..
#When the serie has a limited dependence in long-range time, we can treat the series
# as independent series.

#extremal index \theta, which is equal to the inverse of the limiting mean cluster size


plot(vector_year$maxtemp, type = "l", col = "darkblue",xlab = "Year", ylab = "Maximum winter temperature")
fit1 <- fevd(maxtemp, vector_year, units = "deg C")


#fpot(data$TG, 0.5)


#Block maxima approach
plot(vector_year$maxtemp, type = "l", col = "darkblue", lwd = 1.5, cex.lab = 1.25,xlab = "Year", ylab = " Annual maximum mean temperature")
summary(vector_year$maxtemp)
fit1 <- fevd(maxtemp, vector_year, units = "deg C")

