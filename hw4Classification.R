# Group 4: George Garcia | Marco Perez | Hillary Balli
# Homework 4

#####################
# Libraries and data

library(caret)
library(C50)
library(skimr)
data(churn)

??churn # The outcome variable "churn" is a 2-level nominal factor: Yes or No. 
View(churnTrain)

# Assign variables and combine
xTrain <- churnTrain[, -20]
yTrain <- churnTrain[, 20]
xTest <- churnTest[, -20]
yTest <- churnTest[, 20]
x <- rbind(xTrain, xTest)
y <- c(yTrain,yTest)

#####################
# Visual Exploration

# Basic stats and histogram of dependent variables using skimr
xSkim <- skim_to_wide(x)
# Since there are a couple of skewed variables, preprocess accordingly.

# Plot distribution of outcome factor
barplot(table(y), names.arg = c("Yes", "No"), col = c("blue", "green"), main = "Class Distribution")
# Since the outcome data is imbalanced, use stratified random sampling and subsampling.


###################
# Model Training

# Controlled resampling and subsampling
ctrl <- trainControl(method = "LGOCV", summaryFunction = twoClassSummary, sampling = "smote", 
                     classProbs = TRUE, savePredictions = TRUE)

# Set seed number
seed <- 1234


# Logistic regression
set.seed(seed)
modelLR <- train(x= xTrain,
                 y = yTrain,
                 method = "bayesglm",
                 preProc=c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                 trControl = ctrl)
modelLR # ROC = 0.75, Sens = 0.45, Spec = 0.85


# Linear Discriminant Analysis
set.seed(seed)
modelLDA <- train(x= xTrain[,-(1:5)], # HERE I omitted factor columns, but we need to somehow numerate them. ~George
                  y = yTrain,
                  method = "Linda",
                  preProc=c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                  trControl = ctrl)
modelLDA # ROC = 0.66, Sens = 0.49, Spec = 0.82

