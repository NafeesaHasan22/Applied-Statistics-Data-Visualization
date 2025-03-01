
# Step 1: Install Required Libraries
install.packages("corrplot")
install.packages("stats")
install.packages("caret")
install.packages("glmnet")
install.packages("ggplot2")
install.packages("randomForest")
install.packages("e1071")
install.packages("car")
install.packages("lmtest")
install.packages("readxl")
install.packages("dplyr")
install.packages("tidyr")
install.packages("xgboost")


# Load libraries
library(readxl)
library(glmnet)       
library(randomForest) 
library(e1071)        
library(ggplot2)
library(corrplot)
library(car)
library(caret)
library(stats)
library(lmtest)
library(dplyr)
library(tidyr)
library(xgboost)
# --------------------------------------

# Step 2: Load and Prepare Dataset
concrete <- read_excel("concrete compressive strength.xlsx")

# --------------------------------------

# Step 3: Data Cleaning and Transformation

# Rename columns 
colnames(concrete) <- c(
  "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
  "Superplasticizer", "CoarseAggregate", "FineAggregate",
  "Age", "ConcreteCategory", "ContainsFlyAsh", "CompressiveStrength"
)

# Check for missing values in each column
colSums(is.na(concrete))

# --------------------------------------

# Step 4: Exploratory Data Analysis

names(concrete)
head(concrete)
tail(concrete)
str(concrete)
summary(concrete)

# Check normality with Histogram of compressive strength
ggplot(concrete, aes(x = CompressiveStrength)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  labs(title = "Histogram of Compressive Strength",
       x = "CompressiveStrength", y = "Frequency")

# Boxplots for Outliers
concrete %>%
  gather(key = "Variable", value = "Value", -ConcreteCategory) %>%
  ggplot(aes(x = ConcreteCategory, y = Value)) +
  geom_boxplot(fill = "orange", color = "black") +
  facet_wrap(~Variable, scales = "free") +
  labs(title = "Boxplots for Outlier Detection by ConcreteCategory")

# Shapiro-Wilk test for normality
shapiro_results <- sapply(concrete[, sapply(concrete, is.numeric)],
                          function(x) shapiro.test(x)$p.value)
print(shapiro_results)

# Log-Transformation
concrete$LogCompressiveStrength <- log(concrete$CompressiveStrength)

# --------------------------------------

# Step 5: Correlation Analysis

numeric_columns <- concrete[, sapply(concrete, is.numeric)]
correlation_matrix <- cor(numeric_columns, use = "complete.obs", method = "spearman")
corrplot(correlation_matrix, method = "circle", type = "upper", 
         title = "Spearman Correlation Matrix", tl.cex = 0.8, 
         tl.col = "black", addCoef.col = "black", number.cex = 0.7)

# --------------------------------------

# Step 6: Data-Preprocessing

# Convert categorical columns to one-hot encoding
concrete <- concrete %>%
  mutate(across(where(is.character), ~ as.factor(.))) %>%
  mutate(across(where(is.factor), ~ as.numeric(as.factor(.)) - 1))

# Standardize numeric variables
preprocess_params <- preProcess(concrete, method = c("center", "scale"))
concrete_scaled <- predict(preprocess_params, newdata = concrete)

# Add interaction terms for key variables
concrete_scaled <- concrete_scaled %>%
  mutate(Cement_Age = Cement * Age,
         Water_Age = Water * Age,
         Superplasticizer_Age = Superplasticizer * Age,
         Cement_Water = Cement * Water,
         Water_Cement_Ratio = Water / Cement,  
         WCR_Age = Water_Cement_Ratio * Age)  

# --------------------------------------
# Step 7: Regression Models
# --------------------------------------

# (A) Linear Regression
# --------------------------------------

lm_model <- lm(LogCompressiveStrength ~ ., data = concrete_scaled)
summary(lm_model)

# Multicollinearity Check
vif_values <- vif(lm_model)
print(vif_values)

high_vif_predictor <- names(which.max(vif_values))
print(high_vif_predictor)

concrete_scaled <- concrete_scaled %>% select(-all_of(high_vif_predictor))

# Normality Test: Shapiro-Wilk Test on residuals
shapiro_test <- shapiro.test(residuals(lm_model))
print(shapiro_test)

# Plotting Q-Q Plot for Visual Check of Normality
qqnorm(residuals(lm_model))
qqline(residuals(lm_model), col = "blue", lwd = 2)

# Heteroscedasticity Test: Breusch-Pagan Test
bp_test <- bptest(lm_model)
print(bp_test)

# Visualizing Residuals vs Fitted for Heteroscedasticity Check
plot(lm_model$fitted.values, residuals(lm_model),
     main = "Residuals vs Fitted",
     xlab = "Fitted Values",
     ylab = "Residuals",
     pch = 20, col = "blue")
abline(h = 0, col = "red", lwd = 2)

# --------------------------------------
# (B) Random Forest Regression
# --------------------------------------

# Set seed for reproducibility
set.seed(123)

# Split the data 
train_indices <- createDataPartition(concrete_scaled$LogCompressiveStrength,
                                     p = 0.8, list = FALSE)
train_data <- concrete_scaled[train_indices, ]
test_data <- concrete_scaled[-train_indices, ]

# Train Random Forest on training data
rf_model <- train(
  LogCompressiveStrength ~ ., 
  data = train_data,
  method = "ranger",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = expand.grid(
    mtry = c(3, 4, 5),
    splitrule = "variance",
    min.node.size = c(5, 10, 15)))
  
# Evaluate on the test data
rf_predictions <- predict(rf_model, newdata = test_data)
rf_rmse <- RMSE(rf_predictions, test_data$CompressiveStrength)
rf_r2 <- R2(rf_predictions, test_data$CompressiveStrength)

print(paste("Random Forest Test RMSE:", rf_rmse))
print(paste("Random Forest Test R-squared:", rf_r2))

# Random Forest Residual Diagnostics
rf_residuals <- test_data$CompressiveStrength - rf_predictions

# Residuals vs Predicted
plot(rf_predictions, rf_residuals,
     main = "Residuals vs Predicted (Random Forest)",
     xlab = "Predicted Values", ylab = "Residuals", pch = 20, col = "blue")
abline(h = 0, col = "red", lwd = 2)

# --------------------------------------
# (C) XGBoost Regression
# --------------------------------------

# Prepare Data for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(train_data[, -ncol(train_data)]), 
                      label = train_data$LogCompressiveStrength)

# Define Parameters
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Perform Cross-Validation
xgb_cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 100,
  nfold = 5,
  early_stopping_rounds = 10,
  maximize = FALSE
)
best_nrounds <- xgb_cv$best_iteration

# Train Final Model
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = best_nrounds)

# Evaluate on Test Data
dtest <- xgb.DMatrix(data = as.matrix(test_data[, -ncol(test_data)]))
xgb_predictions <- predict(xgb_model, newdata = dtest)
xgb_rmse <- RMSE(xgb_predictions, test_data$CompressiveStrength)
xgb_r2 <- R2(xgb_predictions, test_data$CompressiveStrength)
print(paste("XGBoost Test RMSE:", xgb_rmse))
print(paste("XGBoost Test R-squared:", xgb_r2))

# XGBoost Residual Diagnostics
xgb_residuals <- test_data$CompressiveStrength - xgb_predictions

# Residuals vs Predicted
plot(xgb_predictions, xgb_residuals,
     main = "Residuals vs Predicted (XGBoost)",
     xlab = "Predicted Values", ylab = "Residuals", pch = 20, col = "blue")
abline(h = 0, col = "red", lwd = 2)

# Shapiro-Wilk Test for normality of residuals
shapiro_test <- shapiro.test(xgb_residuals)
print(shapiro_test)

# Q-Q Plot
qqnorm(xgb_residuals, main = "Q-Q Plot of Residuals (XGBoost)")
qqline(xgb_residuals, col = "blue", lwd = 2)

# Residuals vs Fitted Values
plot(xgb_predictions, xgb_residuals^2,
     main = "Residuals vs Fitted (XGBoost)",
     xlab = "Predicted Values",
     ylab = "Squared Residuals",
     pch = 20, col = "blue")
abline(h = 0, col = "red", lwd = 2)

# --------------------------------------
# (D) Support Vector Regression (SVR)
# --------------------------------------

# Set tuning grid for SVR
svr_grid <- expand.grid(C = c(0.1, 1, 10), sigma = seq(0.01, 0.1, by = 0.01))

# Train SVR model
svr_model <- train(
  LogCompressiveStrength ~ ., 
  data = train_data,
  method = "svmRadial",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = svr_grid
)

# Predictions and evaluation
svr_predictions <- predict(svr_model, newdata = test_data)
svr_rmse <- RMSE(svr_predictions, test_data$CompressiveStrength)
svr_r2 <- R2(svr_predictions, test_data$CompressiveStrength)

print(paste("SVR Test RMSE:", svr_rmse))
print(paste("SVR Test R-squared:", svr_r2))

# Residuals
svr_residuals <- test_data$CompressiveStrength - svr_predictions

# Residuals vs Predicted
plot(svr_predictions, svr_residuals,
     main = "Residuals vs Predicted (SVR)",
     xlab = "Predicted Values", ylab = "Residuals", pch = 20, col = "blue")
abline(h = 0, col = "red", lwd = 2)

# --------------------------------------
# (E) Ridge Regression
# --------------------------------------

# Prepare Data for Ridge Regression
x_train <- model.matrix(LogCompressiveStrength ~ ., data = train_data)[, -1]
y_train <- train_data$LogCompressiveStrength
x_test <- model.matrix(LogCompressiveStrength ~ ., data = test_data)[, -1]
y_test <- test_data$LogCompressiveStrength

# Perform Cross-Validation
ridge_cv <- cv.glmnet(x_train, y_train, alpha = 0)
best_lambda <- ridge_cv$lambda.min

# Train Final Ridge Model
ridge_model <- glmnet(x_train, y_train, alpha = 0, lambda = best_lambda)

# Evaluate on Test Data
ridge_predictions <- predict(ridge_model, newx = x_test)
ridge_rmse <- RMSE(ridge_predictions, test_data$CompressiveStrength)
ridge_r2 <- R2(ridge_predictions, test_data$CompressiveStrength)
print(paste("Ridge Regression Test RMSE:", ridge_rmse))
print(paste("Ridge Regression Test R-squared:", ridge_r2))

# Residuals
ridge_residuals <- y_test - ridge_predictions

# Shapiro-Wilk Test for Normality
shapiro_test <- shapiro.test(ridge_residuals)
print(shapiro_test)

# Q-Q Plot
qqnorm(ridge_residuals, main = "Q-Q Plot of Residuals (Ridge Regression)")
qqline(ridge_residuals, col = "blue", lwd = 2)

# Breusch-Pagan Test for Heteroscedasticity
lm_bp <- lm(ridge_residuals^2 ~ ridge_predictions)
bp_test <- bptest(lm_bp)
print(bp_test)

# Residuals vs Fitted
plot(ridge_predictions, ridge_residuals,
     main = "Residuals vs Fitted (Ridge Regression)",
     xlab = "Fitted Values", ylab = "Residuals", pch = 20, col = "blue")
abline(h = 0, col = "red", lwd = 2)

# --------------------------------------
# Step 8: Hypothesis Testing
# --------------------------------------

# T-Test: Compare compressive strength with and without fly ash
t_test <- t.test(LogCompressiveStrength ~ ContainsFlyAsh, data = concrete)
print(t_test)

# ANOVA: Compare compressive strength across concrete categories
concrete$ConcreteCategory <- as.factor(concrete$ConcreteCategory)
anova_test <- aov(LogCompressiveStrength ~ ConcreteCategory, data = concrete)
summary(anova_test)

# Kruskal-Wallis test
kruskal.test(LogCompressiveStrength ~ ConcreteCategory, data = concrete)

# Chi-Square Test: Association between ContainsFlyAsh and ConcreteCategory
chi_square_test <- chisq.test(table(concrete$ContainsFlyAsh, 
                                    concrete$ConcreteCategory))
print(chi_square_test)

# Correlation Test: Between Cement and Compressive Strength
cor_test <- cor.test(concrete$Cement, concrete$LogCompressiveStrength, 
                     method = "kendall")
print(cor_test)

# Wilcoxon Test: Comparing compressive strength for Coarse vs Fine
wilcox_test <- wilcox.test(LogCompressiveStrength ~ ConcreteCategory, 
                           data = concrete)
print(wilcox_test)






