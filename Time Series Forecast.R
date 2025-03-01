
# Install and load required libraries
install.packages("forecast") # For forecasting models and tools
install.packages("TTR") # For time series smoothing
install.packages("tseries") # For stationarity tests
install.packages("prophet")

library(forecast)
library(TTR)
library(tseries)
library(prophet)

# Load the dataset
data <- read.csv("ADXN.csv")

# Exploration of the Data
names(data)
str(data)
summary(data)
head(data)
tail(data) 

# Data Preprocessing

# Convert Date to proper format and numeric index
  data$Date <- as.Date(data$Date, format = "%Y-%m-%d")
  data$Index <- as.numeric(data$Date)

# Check for missing values
cat("Number of missing values:", sum(is.na(data)), "\n")

# Time Series

# Create a time series object for 'Close' prices
time_series <- ts(data$Close,frequency=12) 

# Summarize the time series
summary(time_series)  

# Plot the time series to visualize trends and patterns
plot.ts(time_series, main="Time Series of Share Prices", 
        ylab="Close Price", xlab="Time")

#  Decomposition

# Decompose the time series into trend, seasonal, and irregular components
decomposed <- decompose(time_series, type="additive") 
plot(decomposed) 

# Extract components
trend <- decomposed$trend
seasonal <- decomposed$seasonal
random <- decomposed$random

# Seasonal and Trend Decomposition using LOESS (STL)
seasonally_adjusted <- seasadj(stl(time_series, s.window = "periodic"))
plot.ts(seasonally_adjusted, main = "Seasonally Adjusted Time Series")

# Stationarity

# ACF and PACF for stationarity check
acf(time_series, main = "ACF of Share Prices")

pacf(time_series, main = "PACF of Share Prices")

# Differencing

# First-order differencing to remove non-stationarity (trend component)
diff_series <- diff(time_series) 
Box.test(diff_series, lag=10, type="Ljung-Box")

# Plot the differenced series
plot.ts(diff_series, main="Differenced Time Series") 

# Plot ACF of the differenced series to confirm stationarity
acf(diff_series, main="ACF of Differenced Time Series")

# Time Series Modeling

# --------------------------------------
# (A) ARIMA Model
# --------------------------------------

# Fit an ARIMA model to the time series
arima_model <- auto.arima(diff_series, stepwise = FALSE, approximation = FALSE)
summary(arima_model)

arima_manual <- Arima(time_series, order = c(1,1,1))
summary(arima_manual)

# Forecast the next 30 time points using the ARIMA model
arima_forecast <- forecast(arima_model, h=30)

# Plot the forecast
plot(arima_forecast, main="ARIMA Forecast")

# Evaluate ARIMA model accuracy
arima_accuracy <- accuracy(arima_forecast)
print(arima_accuracy)

# ARIMA Residual Diagnostics
arima_residuals <- residuals(arima_model)
par(mfrow = c(1, 2))
acf(arima_residuals, main = "ACF of ARIMA Residuals")

hist(arima_residuals, breaks = 20, main = "Histogram of ARIMA Residuals",
     xlab = "Residuals")

# Test for autocorrelation
Box.test(arima_residuals, lag=10, type="Ljung-Box")

# Q-Q plot for normality
qqnorm(arima_residuals)  
qqline(arima_residuals)

# Shapiro-Wilk test for normality
 shapiro.test(arima_residuals)
 
# --------------------------------------
# (B) Holt-Winters Exponential Smoothing
# --------------------------------------

# Fit a Holt-Winters model to the time series
hw_model <- HoltWinters(time_series, beta=TRUE, gamma=TRUE)
summary(hw_model)  

# Forecast the next 30 time points using the Holt-Winters model
hw_forecast <- forecast(hw_model, h=30)

# Plot the forecast
plot(hw_forecast, main="Holt-Winters Forecast")

# Evaluate Holt-Winters model accuracy
hw_accuracy <- accuracy(hw_forecast)
print(hw_accuracy)

# Holt-Winters Residual Diagnostics
hw_residuals <- residuals(hw_forecast)

# Residual diagnostics for Holt-Winters
hw_residuals <- na.omit(residuals(hw_forecast)) 
par(mfrow = c(1, 2))

# Test for autocorrelation
Box.test(hw_residuals, lag=10, type="Ljung-Box")

# Histogram of residuals
hist(hw_residuals, breaks = 20, main = "Histogram of Holt-Winters Residuals", 
     xlab = "Residuals")

# Q-Q plot for normality
qqnorm(hw_residuals)  
qqline(hw_residuals)

# Shapiro-Wilk test for normality
shapiro.test(hw_residuals)

# --------------------------------------
# (C) Prophet Model
# --------------------------------------

# Prepare data for Prophet
data$ds <- as.Date(data$Date)  
data$y <- data$Close          
data <- data[, c("ds", "y")]  

# Create Prophet object without fitting
prophet_model <- prophet(
  daily.seasonality = TRUE,  
  weekly.seasonality = TRUE, 
  yearly.seasonality = TRUE  
)

# Add custom seasonality (30-day periodic pattern)
prophet_model <- add_seasonality(
  prophet_model,
  name = 'custom',
  period = 30,        
  fourier.order = 3   
)

# Fit the model to your data
prophet_model <- fit.prophet(prophet_model, data)

# Make future dataframe and forecast
future <- make_future_dataframe(prophet_model, periods = 30)
forecast <- predict(prophet_model, future)

# Plot the forecast and components
plot(prophet_model, forecast)
prophet_plot_components(prophet_model, forecast)

# Residual diagnostics for Prophet
actuals <- data$y
predicted <- forecast$yhat[1:length(actuals)]
residuals <- actuals - predicted
par(mfrow = c(1, 2))
acf(residuals, main = "ACF of Prophet Residuals")

hist(residuals, breaks = 20, main = "Histogram of Prophet Residuals", 
     xlab = "Residuals")

Box.test(residuals, lag = 10, type = "Ljung-Box")


