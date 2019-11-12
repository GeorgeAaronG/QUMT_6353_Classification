# Group 4: George Garcia | Marco Perez | Hillary Balli
# Homework 4


# 1) Access the 'churn' data set and implement specified algorithms

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
modelLR <- train(x = xTrain,
                 y = yTrain,
                 method = "bayesglm",
                 preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                 trControl = ctrl)
modelLR # ROC = 0.75, Sens = 0.45, Spec = 0.85



# Linear Discriminant Analysis
set.seed(seed)
modelLDA <- train(x = xTrain[,-(1:5)], # HERE I omitted factor columns, but we need to somehow numerate them. ~George
                  y = yTrain,
                  method = "Linda",
                  preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                  trControl = ctrl)
modelLDA # ROC = 0.66, Sens = 0.49, Spec = 0.82



# Partial Least Squares Discriminant Analysis
set.seed(seed)
modelPLS <- train(x = xTrain[, -(1:5)],
                  y = yTrain,
                  method = "pls",
                  preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                  metric = "Accuracy",
                  trControl = ctrl)
modelPLS # ROC = 0.72, Sens = 0,53, Spec = 0.77



# NOTE: I did not get this one to work ~George
# Penalized Model: Logistic Regression
gridGLMN <- expand.grid(alpha = c(0, .1, .2, .4), lambda = seq(.01, .2, length = 10))

set.seed(seed)
modelLR2 <- train(x = xTrain,
                  y = yTrain,
                  method = "glmnet",
                  preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                  tuneGrid = gridGLMN,
                  metric = "Accuracy",
                  trControl = ctrl,
                  family = "multinomial")
modelLR2



# Penalized Model: LDA
library(sparseLDA)
set.seed(seed)
modelLDA <- sda(x = xTrain[, -(1:5)], y = yTrain, lambda = 0.01, stop = -6)
modelLDA



# Nearest Shrunken Centroids
library(pamr)
gridNSC <- data.frame(threshold = seq(0, 4, by=0.1))
set.seed(seed)
modelNSC <- train(x = xTrain[,-(1:5)], y = yTrain, method = "pam",
                preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                tuneGrid = gridNSC,
                metric = "Accuracy",
                trControl = ctrl)
modelNSC # ROC = 0.71, Sens = 0.09, Spec = 0.99



# Nonlinear Discriminant Analysis
gridMDA <- expand.grid(subclasses = 5)
set.seed(seed)
modelMDA <- train(x = xTrain, y = yTrain, method = "mda",
                preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                metric = "Accuracy",
                tuneGrid = gridMDA,
                trControl = ctrl)
modelMDA # ROC = 0.73, Sens = 0.31, Spec = 0.91



# Support Vector Machines
library(MASS)
library(kernlab)
gridSVM <- expand.grid(sigma = c(0.01,0.05,0.1), C = c(1))
set.seed(seed)
modelSVM <- train(x = xTrain[,-(1:5)], 
                   y = yTrain,
                   method = "svmRadial",
                   metric = "Accuracy",
                   preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                   tuneGrid = gridSVM,
                   fit = FALSE,
                   trControl = ctrl)
modelSVM # ROC = 0.79, Sens = 0.59, Spec = 0.86



# K-Nearest Neighbors
gridKNN <- expand.grid(k = c(7,9,11,13,15,17,19))
set.seed(seed)
modelKNN <- train(x = xTrain[,-(1:5)], 
                y = yTrain,
                method = "knn",
                metric = "Accuracy",
                preProc= c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                tuneGrid = gridKNN,
                trControl = ctrl)
modelKNN # ROC = 0.77, Sens = 0.61, Spec = 0.80



# Naive Bayes
library(klaR)
set.seed(seed)
modelNB <- train( x = xTrain[,-(1:5)], 
                y = yTrain,
                method = "nb",
                preProc=c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                #useKernel=TRUE,
                #fL=2,
                trControl = ctrl)
modelNB # ROC = 0.76, Sens = 0.53, Spec = 0.85




# 2) Implement xgboost on Boston Housing data set.

###########################
# Load Packages and Data

library(mlbench)
library(xgboost)
library(plyr)
data(BostonHousing)

# Split data into training and test sets with stratified random sampling
x2 <- subset(BostonHousing, select = -medv)
y2 <- subset(BostonHousing, select = medv)

set.seed(seed)
trainingRows <- createDataPartition(BostonHousing$medv,
                                    p = 0.7,
                                    list = FALSE)

x2Test <- x2[-trainingRows,]
y2Test <- y2[-trainingRows,]
x2Train <- x2[trainingRows,]
y2Train <- y2[trainingRows,]




##########################
# eXtreme Gradient Boost Model Training

# 10-fold cross validation control
control2 <- trainControl(method = "repeatedcv", number = 10, repeats = 10) 

# eXtreme Gradient Boosting Tree
set.seed(seed)
modelXGBT <- train(x = x2Train[,-4], y = y2Train, method = "xgbTree",
                preProc=c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                trControl = control2)
modelXGBT #RMSE = 3.54


# eXtreme Gradient Boosting
set.seed(seed)
modelXGBD <- train(x = x2Train[,-4], y = y2Train, method = "xgbDART", 
                   preProc=c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                   trControl = control2)
modelXGBD #RMSE = 


set.seed(seed)
modelXGBL <- train(x = x2Train[,-4], y = y2Train, method = "xgbLinear",
                   preProc=c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                   trControl = control2)
modelXGBL #RMSE = 3.65, R^2 = 0.85, MAE = 2.47



######################
# Caret Ensembles

# Use caretEnsemble package to see if it is possible to make predictions better.
library(caretEnsemble)

# Define training control
control3 <- trainControl(method = "repeatedcv", number = 10, repeats = 3, 
                        index = createResample(y2Train, 10),
                        savePredictions="final")


# Train a list of models with caretList()
listOfModels <- c('xgbTree', 'xgbLinear')
set.seed(seed)
modelList <- caretList(x = x2Train[,-4], y = y2Train, 
                    trControl = control3,
                    preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
                    methodList = listOfModels)
results <- resamples(modelList)
summary(results)
dotplot(results)
modelCor(results) # xgbTree and xgbLinear have correlation value = 0.68

# Combine the models with caretEnsemble()
set.seed(seed)
modelEnsemble <- caretEnsemble(modelList, metric = "RMSE",
                                 trControl = trainControl(number = 2))

summary(modelEnsemble) # RMSE = 3.60

# Therefore, an ensemble of xgbTree and xgbLinear does not perform better than xgbTree alone.