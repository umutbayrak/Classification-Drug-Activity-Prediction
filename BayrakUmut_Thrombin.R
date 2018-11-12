## Author: Umut Bayrak
## Task: Big Data Project, Drug Activity Prediction
## Version: 3.4.1 (2017-06-30)
## Last update: 10.06.2018 22:15
################################

## Below code is used for drug activity prediction
## Since the dataset is highly imbalanced the test error is estimated
## via k-fold cross-validation.
## For each model, a function is built with the format of k.fold."model"."set"
## Within these functions, model is built on folds, tested with predictions
## And Evaluation metrics are stored in a matrix. 
## Then, this cross validation is repeated 50 times to get a distribution of 
## Evaluation metrics, at the end these are plotted and model selection is done.


###### LIBRARIES USED ######

# For reading in large data sets:
library(data.table)

# For feature selection:
library(mlbench)
library(caret)

library(FactoMineR)

library(dplyr)

# For sparse solutions, regularization, logistic regression
library(glmnet)

# For sparse matrix conversion
library(Matrix)

#For plotting
library(ggplot2)

# For building models
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(randomForest)
library(pls)
library(e1071)
library(DMwR)

## READING IN DATA
thrombin.train <- fread("C:/Users/Umut/MaStat17/Thrombin/Thrombin.train.txt")
thrombin.test <- fread("C:/Users/Umut/MaStat17/Thrombin/Thrombin.test.txt")


## Imbalanced dataset with IR: 44:1

#table(thrombin.train$V1)

#round(prop.table(table(thrombin.train$V1)),2)
#    A    I 
# 0.02 0.98

## Labels are stored in "labels"
labels <- thrombin.train[, 1]

labels <- as.factor(ifelse(labels$V1 == "A", 1, 0))


## Transforming into a data matrix
thrombin.matrix <- data.matrix(thrombin.train[,-1], 
                               rownames.force = NA)

## Converting the main matrix in to a sparse matrix
thrombin.sparse <- Matrix(thrombin.matrix, sparse = TRUE)




##################################################
########## CLASSIFICATION ALGORITHMS #############
##################################################


##### PENALISED REGRESSION WITH ORIGINAL DATA SET
#################################################

## Grid search for best penalization parameters
set.seed(1991)

alpha.grid <- seq(0.5, 1, length = 50)
min.lambda <- rep(NA, 1, 50)
train.error <- rep(NA, 1, 50)


## Searching for the best parameters
for(a in 1:length(alpha.grid))
{
  set.seed(170323)
  pros.search.ela.cv <- cv.glmnet(x = thrombin.sparse ,
                                  y = labels,
                                  alpha = alpha.grid[a],
                                  family = "binomial")
  
  min.lambda[a] <- pros.search.ela.cv$lambda.min
  train.error[a] <- min(pros.search.ela.cv$cvm) #the mean cross-validated error
  
}

## Storing alpha, lambda, and training error in search.ela
search.ela <- cbind(alpha.grid, min.lambda, train.error)
colnames(search.ela) <- c("alpha", "lambda", "Training error")


## The minimum training error is stored in selected
selected <- search.ela[which.min(train.error),]

#        alpha         lambda     Training error 
#     0.50000000     0.01225369     0.09659640


## Building the penaliszd logistic regression model with the selected values

ela.model <- glmnet(x = thrombin.sparse,
                    y = labels,
                    alpha = 0.5,
                    family = "binomial")


## 5-fold cross validation for estimation of evaluation metrics

k.folds <- function(k) {
  folds <- createFolds(labels, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    ela.model <- glmnet(x = thrombin.sparse[folds[[i]],],
                        y = labels[folds[[i]]],
                        alpha = 0.5,
                        family = "binomial")
    
    predictions <- predict(object = ela.model, 
                           newx = thrombin.sparse[-folds[[i]],],
                           s = 0.01225369,
                           type = "class")
    
    accuracies.dt <- c(accuracies.dt, 
                       confusionMatrix(as.factor(predictions), 
                                       labels[-folds[[i]]])$byClass["Balanced Accuracy"])
    
    kappa.dt <- c(kappa.dt,
                  confusionMatrix(as.factor(predictions), 
                                  labels[-folds[[i]]])$overall["Kappa"])
  }
  cbind(accuracies.dt, kappa.dt)
}

ptm <- proc.time()
set.seed(567)
accuracies.dt <- c()
kappa.dt <- c()
eval.glm <- k.folds(5)
proc.time() - ptm


#### REPEATED 5 - FOLD CROSS VALIDATION x50 times

set.seed(567)
v <- c()
v <- replicate(50, k.folds(5))
eval.glm <- c()

## Storing the metrics in a matrix
eval.glm <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.glm) <- c("Balanced Accuracy", "Kappa")




## FEATURE SELECTION WITH ELASTIC NET
######################################


dim(ela.model$beta)
coef.sum <- rowSums(ela.model$beta)

## All genes that survived the elastic net
selected.ela <- names(coef.sum[coef.sum!=0])

## Top 5 genes that affect the prediction
coef.sum[coef.sum %in% tail(sort(abs(coef.sum)), 5)]

#   V67657    V79652    V91840   V111571   V139071 
# 99.99785 160.37569 149.65848 189.64391 136.83240 



##### EXTREME GRADIENT BOOSTING (XGB) ALGORITHM
#######################################################

## Misclassification error function is defined to assess the predictions in CV

MISERR <- function(observedY, predictedY, c=0.5){
  ypred <- ifelse(predictedY > c, 1, 0)
  tab <- table(observedY, ypred)
  miserr <- 1 - sum(diag(tab)) / sum(tab)
  return(miserr)
}


## 5-fold cross validation for estimation of balanced accuracy

label.numeric <- as.numeric(labels)
label.numeric <- ifelse(label.numeric == 1, 0, 1)


## k-fold CV function is defined 

k.folds.xgb <- function(k) {
  folds <- createFolds(labels, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    
    ## Training model with folds
    xgb <- xgboost(data = thrombin.sparse[folds[[i]], ], 
                   label = label.numeric[folds[[i]]], 
                   eta = 0.1,
                   max_depth = 15, 
                   nround=25, 
                   subsample = 0.5,
                   colsample_bytree = 0.5,
                   seed = 1,
                   eval_metric = "auc",
                   objective = "binary:logistic",
                   nthread = 3)
    
    ## Test predictions with remaining fold
    xgb.pred <- predict(xgb, 
                        newdata = thrombin.sparse[-folds[[i]], ],
                        type = "response")
    
    
    ## Specifying the cut-off value here
    c_miss_error.xgb <- c()
    count <- 0
    for(c in xgb.pred){
      count <- count + 1
      c_miss_error.xgb[count] <- MISERR(labels[-folds[[i]]], xgb.pred, c = c)
    }
    
    ## The value that gives the minimum misclassification error is set
    c.xgb <- xgb.pred[c_miss_error.xgb == min(c_miss_error.xgb)]
    
    
    ## Prediction are categorized
    xgb.pred.values <- ifelse(xgb.pred > c.xgb, 1, 0)
    
    
    ## Balanced Accuracy is stored from Confusion Matrix
    accuracies.xgb <- c(accuracies.xgb, 
                       confusionMatrix(as.factor(xgb.pred.values), 
                                       labels[-folds[[i]]])$byClass["Balanced Accuracy"])
    
    ## Kappa is stored from Confusion Matrix
    kappa.xgb <- c(kappa.xgb,
                   confusionMatrix(as.factor(xgb.pred.values), 
                                   labels[-folds[[i]]])$overall["Kappa"])
  }
  cbind(accuracies.xgb, kappa.xgb)
}


## proc.time() is used to measure the time spent on cross validation

## Doing the 5-fold Cross Validation once.
ptm <- proc.time()
set.seed(567)
accuracies.xgb <- c()
kappa.xgb <- c()
eval.xgb <- c()
eval.xgb <- k.folds.xgb(5)
proc.time() - ptm
eval.xgb


#### REPEATED 5 - FOLD CROSS VALIDATION x50 times

set.seed(567)
v <- c()
v <- replicate(50, k.folds.xgb(5))
eval.xgb <- c()

eval.xgb <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.xgb) <- c("Balanced Accuracy", "Kappa")





##### FEATURE SELECTION WITH XGB
#########################################

## Building the model here

xgb <- xgboost(data = thrombin.sparse, 
               label = label.numeric, 
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "auc",
               objective = "binary:logistic",
               nthread = 3)

## Important Variables are extracted

names <- dimnames(thrombin.sparse)[[2]]
importance_matrix <- xgb.importance(names, model = xgb)
xgb.plot.importance(importance_matrix[1:20,])

#    Feature         Gain        Cover Frequency
# 1:  V79652 0.0553837618 0.0328573061     0.024
# 2:  V16795 0.0501966873 0.0240815338     0.016
# 3:  V29370 0.0488166709 0.0170812588     0.008
# 4:  V90407 0.0450419637 0.0218812287     0.016
# 5:  V27152 0.0424622068 0.0162613367     0.008

selected.xgb <- importance_matrix$Feature



######### SUPPORT VECTOR MACHINES
########################################################


## Tuning the SVM model for the best parameters

## NOTE: Takes ten minutes to run DO NOT run if not necessary
## Results are listed below with obj$parameters

ptm <- proc.time()

obj <- tune(method = "svm",
                train.x = thrombin.sparse,
                train.y = labels,
                ranges = list(gamma = 10^(-6:-1), cost = 2^(2:8)),
                tunecontrol = tune.control(sampling = "fix"))
proc.time() - ptm



obj$best.parameters
#    gamma cost
#    1e-04   4

## k-fold CV function is defined

k.folds.svm<- function(k) {
  folds <- createFolds(labels, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    # Model Building
    svm.fit <- svm(y=labels[folds[[i]]],
                   x = thrombin.sparse[folds[[i]],],
                   cost = 4,
                   gamma = 1e-04,
                   type = "C",
                   kernel = "linear")
    
    # Predictions
    svm.pred <- predict(svm.fit, 
                            newdata = thrombin.sparse[-folds[[i]], ],
                            type = "class")
    
    # Evaluations
    accuracies.svm <- c(accuracies.svm, 
                            confusionMatrix(as.factor(svm.pred), 
                                            labels[-folds[[i]]])$byClass["Balanced Accuracy"])
    
    kappa.svm <- c(kappa.svm,
                       confusionMatrix(as.factor(svm.pred), 
                                       labels[-folds[[i]]])$overall["Kappa"])
  }
  cbind(accuracies.svm, kappa.svm)
}

## To measure the time passed for 5-folds cross validation
ptm <- proc.time()

set.seed(567)
accuracies.svm <- c()
kappa.svm <- c()
eval.svm <- c()
eval.svm <- k.folds.svm(5)

proc.time() - ptm


#### REPEATED 5 - FOLD CROSS VALIDATION x50 times

set.seed(567)
v <- c()
v <- replicate(50, k.folds.svm(5))
eval.svm <- c()

eval.svm <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.svm) <- c("Balanced Accuracy", "Kappa")



############# FEATURE SELECTION WITH LASSO ##################
#############################################################

lasso.fit <- glmnet(y = labels,
                    x = thrombin.sparse,
                    family = "binomial")


## The coefficients differ from 0 are stored.
lasso.sum <- rowSums(lasso.fit$beta)
selected.lasso <- names(lasso.sum[lasso.sum != 0])





#############################################################
########  ORIGINAL DATASETS FILE WITH SELECTED FEATURES (NO SPARSE)
#############################################################

thrombin.ela <- data.frame(thrombin.train[ ,..selected.ela])
thrombin.xgb <- data.frame(thrombin.train[ ,..selected.xgb])
thrombin.lasso <- data.frame(thrombin.train[ ,..selected.lasso])

thrombin.ela$class <- labels
thrombin.xgb$class <- labels
thrombin.lasso$class <- labels

test.ela <- data.frame(thrombin.test[ ,..selected.ela])
test.xgb <- data.frame(thrombin.test[ ,..selected.xgb])
test.lasso <- data.frame(thrombin.test[ ,..selected.lasso])

prop.table(table(labels))*100
prop.table(table(smote.ela$class))*100


############################################################
##### SVM WITH XGB FEATURES - CLASS WEIGHTS

## With tune I will find the best cost and gamma
## Tuning works better with sparse matrices 

all.xgb <- thrombin.sparse[ ,selected.xgb]
# all.xgb$class <- as.numeric(all.xgb$class)
# all.xgb$class <- ifelse(all.xgb$class == 1, 0, 1)

ptm <- proc.time()

obj.xgb <- tune(method = "svm",
                train.x = all.xgb,
                train.y = labels,
                ranges = list(gamma = 10^(-6:-1), cost = 2^(2:8)),
                tunecontrol = tune.control(sampling = "fix"))
proc.time() - ptm

obj.xgb$best.parameters
#    gamma cost
#     0.01   16

## k-fold cross validation function is defined:

k.folds.svm.xgb <- function(k) {
  folds <- createFolds(labels, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    
    # Model Building
    svm.fit <- svm(thrombin.xgb$class[folds[[i]]]~.,
                   data = thrombin.xgb[folds[[i]], -which(names(thrombin.xgb) == "class")],
                   cost = 16,
                   gamma = 0.01,
                   type = "C",
                   kernel = "linear",
                   class.weights = c("0" = 44, "1" = 1))
    
    # Prediction
    svm.xgb.pred <- predict(svm.fit, 
                            newdata = thrombin.xgb[-folds[[i]], -which(names(thrombin.xgb) == "class") ],
                            type = "class")
    
    # Evaluation
    accuracies.svm.xgb <- c(accuracies.svm.xgb, 
                            confusionMatrix(as.factor(svm.xgb.pred), 
                                            labels[-folds[[i]]])$byClass["Balanced Accuracy"])
    
    kappa.svm.xgb <- c(kappa.svm.xgb,
                       confusionMatrix(as.factor(svm.xgb.pred), 
                                       labels[-folds[[i]]])$overall["Kappa"])
  }
  cbind(accuracies.svm.xgb, kappa.svm.xgb)
}

## To measure the time passed for 5-folds cross validation
ptm <- proc.time()

set.seed(567)
accuracies.svm.xgb <- c()
kappa.svm.xgb <- c()
eval.svm.xgb <- c()
eval.svm.xgb <- k.folds.svm.xgb(5)

proc.time() - ptm

eval.svm.xgb

#### REPEATED 5 - FOLD CROSS VALIDATION x50 times

set.seed(567)
v <- c()
v <- replicate(50, k.folds.svm.xgb(5))
eval.svm.xgb <- c()

eval.svm.xgb <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.svm.xgb) <- c("Balanced Accuracy", "Kappa")



#################################################################
### 5 - FOLD CV FOR SVM WITH ELASTIC NET FEATURES - CLASS WEIGHTS
#################################################################


## Tune works better with sparse matrices, for best parameters

all.ela <- thrombin.sparse[ ,selected.ela]
# all.ela$class <- as.numeric(all.ela$class)
# all.ela$class <- ifelse(all.ela$class == 1, 0, 1)

ptm <- proc.time()

obj.ela <- tune(method = "svm",
                train.x = all.ela,
                train.y = labels,
                ranges = list(gamma = 10^(-6:-1), cost = 2^(2:8)),
                tunecontrol = tune.control(sampling = "fix"))
proc.time() - ptm

obj.ela$best.parameters
#    gamma cost
#     0.01   4


## Defining k-fol CVd function:

k.folds.svm.ela <- function(k) {
  folds <- createFolds(labels, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    
    # Model Building
    svm.fit <- svm(thrombin.ela$class[folds[[i]]]~.,
                   data = thrombin.ela[folds[[i]], -which(names(thrombin.ela) == "class")],
                   cost = 4,
                   gamma = 0.01,
                   type = "C",
                   kernel = "linear",
                   class.weights = c("0" = 44, "1" = 1))
    
    # Prediction
    svm.ela.pred <- predict(svm.fit, 
                            newdata = thrombin.ela[-folds[[i]], -which(names(thrombin.ela) == "class") ],
                            type = "class")
    
    # Evaluation
    accuracies.svm.ela <- c(accuracies.svm.ela, 
                            confusionMatrix(as.factor(svm.ela.pred), 
                                            labels[-folds[[i]]])$byClass["Balanced Accuracy"])
    
    kappa.svm.ela <- c(kappa.svm.ela,
                       confusionMatrix(as.factor(svm.ela.pred), 
                                       labels[-folds[[i]]])$overall["Kappa"])
  }
  cbind(accuracies.svm.ela, kappa.svm.ela)
}

## To measure the time passed for 5-folds cross validation
ptm <- proc.time()

set.seed(567)
accuracies.svm.ela <- c()
kappa.svm.ela <- c()
eval.svm.ela <- c()
eval.svm.ela <- k.folds.svm.ela(5)

proc.time() - ptm

eval.svm.ela

#### REPEATED 5 - FOLD CROSS VALIDATION x50 times

set.seed(567)
v <- c()
v <- replicate(50, k.folds.svm.ela(5))
eval.svm.ela <- c()

eval.svm.ela <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.svm.ela) <- c("Balanced Accuracy", "Kappa")



###########################################################
### 5 - FOLD CV FOR SVM WITH LASSO FEATURES - CLASS WEIGHTS


## Tuning for the best parameters, it works better with sparse matrices

all.lasso <- thrombin.sparse[ ,selected.lasso]
# all.lasso$class <- as.numeric(all.lasso$class)
# all.lasso$class <- ifelse(all.lasso$class == 1, 0, 1)

ptm <- proc.time()

obj.lasso <- tune(method = "svm",
                  train.x = all.lasso,
                  train.y = labels,
                  ranges = list(gamma = 10^(-6:-1), cost = 2^(2:8)),
                  tunecontrol = tune.control(sampling = "fix"))
proc.time() - ptm

obj.lasso$best.parameters
#    gamma cost
#     0.01   16


## k-fold CV function is defined
k.folds.svm.lasso <- function(k) {
  folds <- createFolds(labels, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    
    # Model Building
    svm.fit <- svm(thrombin.lasso$class[folds[[i]]]~.,
                   data = thrombin.lasso[folds[[i]], -which(names(thrombin.lasso) == "class")],
                   cost = 16,
                   gamma = 0.01,
                   type = "C",
                   kernel = "linear",
                   class.weights = c("0" = 44, "1" = 1))
    
    # Prediction
    svm.lasso.pred <- predict(svm.fit, 
                            newdata = thrombin.lasso[-folds[[i]], -which(names(thrombin.lasso) == "class") ],
                            type = "class")
    
    # Evaluation
    accuracies.svm.lasso <- c(accuracies.svm.lasso, 
                            confusionMatrix(as.factor(svm.lasso.pred), 
                                            labels[-folds[[i]]])$byClass["Balanced Accuracy"])
    
    kappa.svm.lasso <- c(kappa.svm.lasso,
                         confusionMatrix(as.factor(svm.lasso.pred), 
                                         labels[-folds[[i]]])$overall["Kappa"])
  }
  cbind(accuracies.svm.lasso, kappa.svm.lasso)
}

## To measure the time passed for 5-folds cross validation
ptm <- proc.time()

set.seed(567)
accuracies.svm.lasso <- c()
kappa.svm.lasso <- c()
eval.svm.lasso <- c()
eval.svm.lasso <- k.folds.svm.lasso(5)

proc.time() - ptm

eval.svm.lasso

#### REPEATED 5 - FOLD CROSS VALIDATION x50 times

set.seed(567)
v <- c()
v <- replicate(50, k.folds.svm.lasso(5))
eval.svm.lasso <- c()

eval.svm.lasso <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.svm.lasso) <- c("Balanced Accuracy", "Kappa")


####################################################################
###### SUPPORT VECTOR MACHINES WITH SMOTE
####################################################################

## For the parameters SMOTE is applied to the feature selected datasets.

smote.ela <- SMOTE(class ~ ., 
                   thrombin.ela, perc.over = 1000, perc.under=100)
smote.xgb <- SMOTE(class ~ ., 
                   thrombin.xgb, perc.over = 1000, perc.under=100)
smote.lasso <- SMOTE(class ~ ., 
                     thrombin.lasso, perc.over = 1000, perc.under=100)





#### TRAINING SET WITH LASSO SELECTED FEATURES

##### 5- FOLDS CV WITH SMOTE

ptm <- proc.time()

obj.smote.lasso <- tune(method = "svm",
                        train.x = smote.lasso[, -which(names(smote.lasso) == "class")],
                        train.y = smote.lasso$class,
                        ranges = list(gamma = 10^(-6:-1), cost = 2^(2:8)),
                        tunecontrol = tune.control(sampling = "fix"))
proc.time() - ptm

obj.smote.lasso$best.parameters
#    gamma cost
#    0.001   64

## k-fold CV function is defined here

k.folds.svm.lasso.smote <- function(k) {
  folds <- createFolds(thrombin.lasso$class, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    
    ## SMOTE is applied only to the training folds
    x <- SMOTE(class ~ ., thrombin.lasso[folds[[i]],], perc.over = 1000, perc.under=100)
    
    ## Model building
    svm.fit <- svm(x$class~.,
                   data = x[, -which(names(x) == "class")],
                   cost = 64,
                   gamma = 0.001,
                   type = "C",
                   kernel = "linear")
    
    ## Prediction
    smote.svm.lasso.pred <- predict(svm.fit, 
                                    newdata = thrombin.lasso[-folds[[i]], -which(names(thrombin.lasso) == "class") ],
                                    type = "class")
    
    ## Evaluation
    accuracies.svm.lasso.smote <- c(accuracies.svm.lasso.smote, 
                                     confusionMatrix(as.factor(smote.svm.lasso.pred), 
                                                     as.factor(thrombin.lasso$class[-folds[[i]]]))$byClass["Balanced Accuracy"])
    
    kappa.svm.lasso.smote <- c(kappa.svm.lasso.smote,
                         confusionMatrix(as.factor(smote.svm.lasso.pred), 
                                         thrombin.lasso$class[-folds[[i]]])$overall["Kappa"])
  }
  cbind(accuracies.svm.lasso.smote, kappa.svm.lasso.smote)
}


## To measure the time passed for 5-folds cross validation
ptm <- proc.time()

set.seed(567)
accuracies.svm.lasso.smote <- c()
kappa.svm.lasso.smote <- c()
eval.svm.lasso.smote <- c()
eval.svm.lasso.smote <- k.folds.svm.lasso.smote(5)

proc.time() - ptm

eval.svm.lasso.smote

#### REPEATED 5 - FOLD CROSS VALIDATION x50 times

set.seed(567)
v <- c()
v <- replicate(50, k.folds.svm.lasso.smote(5))
eval.svm.lasso.smote <- c()

eval.svm.lasso.smote <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.svm.lasso.smote) <- c("Balanced Accuracy", "Kappa")




#### TRAINING SET WITH ELASTIC NET SELECTED FEATURES
####################################################

##### 5- FOLDS CV WITH SMOTE


## Tune is done for best parameters selection
ptm <- proc.time()

obj.smote.ela <- tune(method = "svm",
                        train.x = smote.ela[, -which(names(smote.ela) == "class")],
                        train.y = smote.ela$class,
                        ranges = list(gamma = 10^(-6:-1), cost = 2^(2:8)),
                        tunecontrol = tune.control(sampling = "fix"))
proc.time() - ptm

obj.smote.lasso$best.parameters
#    gamma cost
#    0.001   64

## k-fold CV function is defined here

k.folds.svm.ela.smote <- function(k) {
  folds <- createFolds(thrombin.ela$class, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    
    ## SMOTE is applied only to the training folds
    x <- SMOTE(class ~ ., 
               thrombin.ela[folds[[i]],], perc.over = 1000, perc.under=100)
    
    ## Model Building
    svm.fit <- svm(x$class~.,
                   data = x[, -which(names(x) == "class")],
                   cost = 64,
                   gamma = 0.001,
                   type = "C",
                   kernel = "linear")
    
    ## Prediction
    smote.svm.ela.pred <- predict(svm.fit, 
                                  newdata = thrombin.ela[-folds[[i]], -which(names(thrombin.ela) == "class") ],
                                  type = "class")
    
    ## Evaluation
    accuracies.svm.ela.smote <- c(accuracies.svm.ela.smote, 
                                  confusionMatrix(as.factor(smote.svm.ela.pred), 
                                                  as.factor(thrombin.ela$class[-folds[[i]]]))$byClass["Balanced Accuracy"])
    
    kappa.svm.ela.smote <- c(kappa.svm.ela.smote,
                             confusionMatrix(as.factor(smote.svm.ela.pred), 
                                             thrombin.ela$class[-folds[[i]]])$overall["Kappa"])
  }
  cbind(accuracies.svm.ela.smote, kappa.svm.ela.smote)
}


## To measure the time passed for 5-folds cross validation
ptm <- proc.time()

set.seed(567)
accuracies.svm.ela.smote <- c()
kappa.svm.ela.smote <- c()
eval.svm.ela.smote <- c()
eval.svm.ela.smote <- k.folds.svm.ela.smote(5)

proc.time() - ptm

eval.svm.ela.smote

#### REPEATED 5 - FOLD CROSS VALIDATION x50 times

set.seed(567)
v <- c()
v <- replicate(50, k.folds.svm.ela.smote(5))
eval.svm.ela.smote <- c()

eval.svm.ela.smote <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.svm.ela.smote) <- c("Balanced Accuracy", "Kappa")

##################################
#### XGB SELECTED FEATURES

##### 5- FOLDS CV WITH SMOTE

ptm <- proc.time()

obj.smote.xgb <- tune(method = "svm",
                      train.x = smote.xgb[, -which(names(smote.xgb) == "class")],
                      train.y = smote.xgb$class,
                      ranges = list(gamma = 10^(-6:-1), cost = 2^(2:8)),
                      tunecontrol = tune.control(sampling = "fix"))
proc.time() - ptm

obj.smote.lasso$best.parameters
#    gamma cost
#    0.001   64

## k-fold CV function is defined here

k.folds.svm.xgb.smote <- function(k) {
  folds <- createFolds(thrombin.xgb$class, k = k, list = TRUE, 
                       returnTrain = TRUE)
  for (i in 1:k) {
    
    ## SMOTE is applied only to the training folds
    x <- SMOTE(class ~ ., 
               thrombin.xgb[folds[[i]],], perc.over = 1000, perc.under=100)
    
    ## Model Building
    svm.fit <- svm(x$class~.,
                   data = x[, -which(names(x) == "class")],
                   cost = 64,
                   gamma = 0.001,
                   type = "C",
                   kernel = "linear")
    
    ## Prediction
    smote.svm.xgb.pred <- predict(svm.fit, 
                                  newdata = thrombin.xgb[-folds[[i]], -which(names(thrombin.xgb) == "class") ],
                                  type = "class")
    
    ## Evaluation
    accuracies.svm.xgb.smote <- c(accuracies.svm.xgb.smote, 
                                  confusionMatrix(as.factor(smote.svm.xgb.pred), 
                                                  as.factor(thrombin.xgb$class[-folds[[i]]]))$byClass["Balanced Accuracy"])
    
    kappa.svm.xgb.smote <- c(kappa.svm.xgb.smote,
                             confusionMatrix(as.factor(smote.svm.xgb.pred), 
                                             thrombin.xgb$class[-folds[[i]]])$overall["Kappa"])
  }
  cbind(accuracies.svm.xgb.smote, kappa.svm.xgb.smote)
}

## To measure the time passed for 5-folds cross validation
ptm <- proc.time()

set.seed(567)
accuracies.svm.xgb.smote <- c()
kappa.svm.xgb.smote <- c()
eval.svm.xgb.smote <- c()
eval.svm.xgb.smote <- k.folds.svm.xgb.smote(5)

proc.time() - ptm

eval.svm.xgb.smote

#### REPEATED 5 - FOLD CROSS VALIDATION x50 times

set.seed(567)
v <- c()
v <- replicate(50, k.folds.svm.xgb.smote(5))
eval.svm.xgb.smote <- c()

eval.svm.xgb.smote <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.svm.xgb.smote) <- c("Balanced Accuracy", "Kappa")


#########################################################
####### ELASTIC NET WITH SELECTED FEATURES ##############
#########################################################

## ELASTIC NET, SMOTE, LASSO FEATURES

## 5-fold cross validation for estimation of balanced accuracy and kappa

k.folds.glm.lasso.smote <- function(k) {
  folds <- createFolds(thrombin.lasso$class, k = k, list = TRUE, 
                       returnTrain = TRUE)
  for (i in 1:k) {
    ## SMOTE is applied only to the training folds
    x <- SMOTE(class ~ ., 
               thrombin.lasso[folds[[i]],], perc.over = 1000, perc.under=100)

    ## Model Building
    ela.model <- glmnet(x = data.matrix(x[, -which(names(thrombin.lasso) == "class")]),
                        y = x$class,
                        alpha = 0.5,
                        family = "binomial")
    
    ## Prediction
    predictions <- predict(object = ela.model, 
                           newx = data.matrix(thrombin.lasso[-folds[[i]],-which(names(thrombin.lasso) == "class")], rownames.force = NA),
                           s = 0.01225369,
                           type = "class")
    ## Evaluation
    accuracies.glm.lasso.smote <- c(accuracies.glm.lasso.smote, 
                                  confusionMatrix(as.factor(predictions), 
                                                  as.factor(thrombin.lasso$class[-folds[[i]]]))$byClass["Balanced Accuracy"])
    
    
    kappa.glm.lasso.smote <- c(kappa.glm.lasso.smote, 
                               confusionMatrix(as.factor(predictions), 
                                               thrombin.lasso$class[-folds[[i]]])$overall["Kappa"])
  }
  cbind(accuracies.glm.lasso.smote, kappa.glm.lasso.smote)
}

## 5-fold Cross-Validation

ptm <- proc.time()
set.seed(567)
eval.glm.lasso.smote <- c()
accuracies.glm.lasso.smote <- c()
kappa.glm.lasso.smote <- c()
eval.glm.lasso.smote <- k.folds.glm.lasso.smote(5)
proc.time() - ptm
eval.glm.lasso.smote



#### REPEATED 5 - FOLD CROSS VALIDATION x50 times

set.seed(567)
v <- c()
v <- replicate(50, k.folds.glm.lasso.smote(5))
eval.glm.lasso.smote <- c()

eval.glm.lasso.smote <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.glm.lasso.smote) <- c("Balanced Accuracy", "Kappa")


## ELASTIC NET, SMOTE, ELASTIC NET FEATURES

## 5-fold cross validation for estimation of balanced accuracy and kappa

k.folds.glm.ela.smote <- function(k) {
  folds <- createFolds(thrombin.ela$class, k = k, list = TRUE, 
                       returnTrain = TRUE)
  for (i in 1:k) {
    ## SMOTE is applied only to the training folds
    x <- SMOTE(class ~ ., 
               thrombin.ela[folds[[i]],], perc.over = 1000, perc.under=100)
    
    ## Model Building
    ela.model <- glmnet(x = data.matrix(x[, -which(names(thrombin.ela) == "class")]),
                        y = x$class,
                        alpha = 0.5,
                        family = "binomial")
    
    ## Prediction
    predictions <- predict(object = ela.model, 
                           newx = data.matrix(thrombin.ela[-folds[[i]],-which(names(thrombin.ela) == "class")], rownames.force = NA),
                           s = 0.01225369,
                           type = "class")
    
    ## Evaluation
    accuracies.glm.ela.smote <- c(accuracies.glm.ela.smote, 
                                  confusionMatrix(as.factor(predictions), 
                                                  as.factor(thrombin.ela$class[-folds[[i]]]))$byClass["Balanced Accuracy"])
    
    
    kappa.glm.ela.smote <- c(kappa.glm.ela.smote, 
                             confusionMatrix(as.factor(predictions), 
                                             thrombin.ela$class[-folds[[i]]])$overall["Kappa"])
  }
  cbind(accuracies.glm.ela.smote, kappa.glm.ela.smote)
}


## 5-fold Cross Validation 

ptm <- proc.time()
set.seed(567)
eval.glm.ela.smote <- c()
accuracies.glm.ela.smote <- c()
kappa.glm.ela.smote <- c()
eval.glm.ela.smote <- k.folds.glm.ela.smote(5)
proc.time() - ptm
eval.glm.ela.smote



#### REPEATED 5 - FOLD CROSS VALIDATION x50 times

set.seed(567)
v <- c()
v <- replicate(50, k.folds.glm.ela.smote(5))
eval.glm.ela.smote <- c()

eval.glm.ela.smote <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.glm.ela.smote) <- c("Balanced Accuracy", "Kappa")

## ELASTIC NET, SMOTE, XGB FEATURES

## 5-fold cross validation for estimation of balanced accuracy and kappa

k.folds.glm.xgb.smote <- function(k) {
  folds <- createFolds(thrombin.xgb$class, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    
    ## SMOTE is applied only to the training folds
    x <- SMOTE(class ~ ., thrombin.xgb[folds[[i]],], perc.over = 1000, perc.under=100)
    
    ## Model Building
    xgb.model <- glmnet(x = data.matrix(x[, -which(names(thrombin.xgb) == "class")]),
                        y = x$class,
                        alpha = 0.5,
                        family = "binomial")
    
    ## Prediction
    predictions <- predict(object = xgb.model, 
                           newx = data.matrix(thrombin.xgb[-folds[[i]],-which(names(thrombin.xgb) == "class")], rownames.force = NA),
                           s = 0.01225369,
                           type = "class")
    
    ## Evaluation
    accuracies.glm.xgb.smote <- c(accuracies.glm.xgb.smote, 
                                  confusionMatrix(as.factor(predictions), 
                                                  as.factor(thrombin.xgb$class[-folds[[i]]]))$byClass["Balanced Accuracy"])
    
    
    kappa.glm.xgb.smote <- c(kappa.glm.xgb.smote, 
                             confusionMatrix(as.factor(predictions), 
                                             thrombin.xgb$class[-folds[[i]]])$overall["Kappa"])
  }
  cbind(accuracies.glm.xgb.smote, kappa.glm.xgb.smote)
}

## 5-fold Cross-Validation

ptm <- proc.time()
set.seed(567)
eval.glm.xgb.smote <- c()
accuracies.glm.xgb.smote <- c()
kappa.glm.xgb.smote <- c()
eval.glm.xgb.smote <- k.folds.glm.xgb.smote(5)
proc.time() - ptm
eval.glm.xgb.smote



#### REPEATED 5 - FOLD CROSS VALIDATION x50 times

set.seed(567)
v <- c()
v <- replicate(50, k.folds.glm.xgb.smote(5))
eval.glm.xgb.smote <- c()

eval.glm.xgb.smote <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.glm.xgb.smote) <- c("Balanced Accuracy", "Kappa")


#########################################################
####### XGBOOST WITH SELECTED FEATURES ##################
#########################################################

## XGBOOST, SMOTE, LASSO FEATURES


label.lasso <- as.numeric(smote.lasso$class)
label.lasso <- ifelse(label.lasso == 1, 0, 1)

k.folds.xgb.lasso.smote<- function(k) {
  folds <- createFolds(thrombin.lasso$class, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    ## SMOTE is applied only to the training folds
    x <- SMOTE(class ~ ., 
               thrombin.lasso[folds[[i]],], perc.over = 1000, perc.under=100)
    x$class <- as.numeric(x$class)
    x$class <- ifelse(x$class == 1, 0, 1)
    
    ## Model Building
    xgb <- xgboost(data = data.matrix(x[, -which(names(x) == "class") ], rownames.force = NA), 
                   label = x$class, 
                   eta = 0.1,
                   max_depth = 15, 
                   nround=25, 
                   subsample = 0.5,
                   colsample_bytree = 0.5,
                   seed = 1,
                   eval_metric = "auc",
                   objective = "binary:logistic",
                   nthread = 3)
    
    ## Prediction
    xgb.pred <- predict(xgb, 
                        newdata = data.matrix(thrombin.lasso[-folds[[i]], -which(names(thrombin.lasso) == "class") ], rownames.force = NA),
                        type = "response")
    
    ## Best cut-off value
    c_miss_error.xgb <- c()
    count <- 0
    for(c in xgb.pred){
      count <- count + 1
      c_miss_error.xgb[count] <- MISERR(thrombin.lasso$class[-folds[[i]]], xgb.pred, c = c)
    }
    
    c.xgb <- xgb.pred[c_miss_error.xgb == min(c_miss_error.xgb)]
    
    xgb.pred.values <- ifelse(xgb.pred >= min(c.xgb), 1, 0)
    
    
    ## Evaluation
    accuracies.xgb.lasso.smote <- c(accuracies.xgb.lasso.smote, 
                                  confusionMatrix(as.factor(xgb.pred.values), 
                                                  as.factor(thrombin.lasso$class[-folds[[i]]]))$byClass["Balanced Accuracy"])
    
    kappa.xgb.lasso.smote<- c(kappa.xgb.lasso.smote,
                   confusionMatrix(as.factor(xgb.pred.values), 
                                   as.factor(thrombin.lasso$class[-folds[[i]]]))$overall["Kappa"])
  }
  cbind(accuracies.xgb.lasso.smote, kappa.xgb.lasso.smote)
}


## 5- fold cross validation
ptm <- proc.time()
set.seed(567)
accuracies.xgb.lasso.smote <- c()
kappa.xgb.lasso.smote <- c()
eval.xgb.lasso.smote <- c()
eval.xgb.lasso.smote <- k.folds.xgb.lasso.smote(5)
proc.time() -ptm

eval.xgb.lasso.smote

## REPEATED 5-FOLD CV X50 TIMES

set.seed(567)
v <- c()
v <- replicate(50, k.folds.xgb.lasso.smote(5))
eval.xgb.lasso.smote <- c()

eval.xgb.lasso.smote <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.xgb.lasso.smote) <- c("Balanced Accuracy", "Kappa")


## XGBOOST, SMOTE, ELASTIC NET FEATURES
#########################################
label.ela <- as.numeric(smote.ela$class)
label.ela <- ifelse(label.ela == 1, 0, 1)


k.folds.xgb.ela.smote<- function(k) {
  folds <- createFolds(thrombin.ela$class, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    ## SMOTE is applied only to the training folds
    x <- SMOTE(class ~ ., thrombin.ela[folds[[i]],], perc.over = 1000, perc.under=100)
    x$class <- as.numeric(x$class)
    x$class <- ifelse(x$class == 1, 0, 1)
    
    ## Model Building
    xgb <- xgboost(data = data.matrix(x[, -which(names(x) == "class") ], rownames.force = NA), 
                   label = x$class, 
                   eta = 0.1,
                   max_depth = 15, 
                   nround=25, 
                   subsample = 0.5,
                   colsample_bytree = 0.5,
                   seed = 1,
                   eval_metric = "auc",
                   objective = "binary:logistic",
                   nthread = 3)
    
    ## Prediction
    xgb.pred <- predict(xgb, 
                        newdata = data.matrix(thrombin.ela[-folds[[i]], -which(names(thrombin.ela) == "class") ], rownames.force = NA),
                        type = "response")
    
    ## Best cut-off value
    c_miss_error.xgb <- c()
    count <- 0
    for(c in xgb.pred){
      count <- count + 1
      c_miss_error.xgb[count] <- MISERR(thrombin.ela$class[-folds[[i]]], xgb.pred, c = c)
    }
    
    c.xgb <- xgb.pred[c_miss_error.xgb == min(c_miss_error.xgb)]
    
    xgb.pred.values <- ifelse(xgb.pred >= min(c.xgb), 1, 0)
    
    
    ## Evaluation
    accuracies.xgb.ela.smote <- c(accuracies.xgb.ela.smote, 
                                  confusionMatrix(as.factor(xgb.pred.values), 
                                                  as.factor(thrombin.ela$class[-folds[[i]]]))$byClass["Balanced Accuracy"])
    
    kappa.xgb.ela.smote<- c(kappa.xgb.ela.smote,
                            confusionMatrix(as.factor(xgb.pred.values), 
                                            as.factor(thrombin.ela$class[-folds[[i]]]))$overall["Kappa"])
  }
  cbind(accuracies.xgb.ela.smote, kappa.xgb.ela.smote)
}

## 5-fold CV

ptm <- proc.time()
set.seed(567)
accuracies.xgb.ela.smote <- c()
kappa.xgb.ela.smote <- c()
eval.xgb.ela.smote <- c()
eval.xgb.ela.smote <- k.folds.xgb.ela.smote(5)
proc.time() - ptm

eval.xgb.ela.smote

## REPEATED 5-FOLD CV X50 TIMES

set.seed(567)
v <- c()
v <- replicate(50, k.folds.xgb.ela.smote(5))
eval.xgb.ela.smote <- c()

eval.xgb.ela.smote <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.xgb.ela.smote) <- c("Balanced Accuracy", "Kappa")



## XGBOOST, SMOTE, XGB FEATURES
###############################

label.xgb <- as.numeric(smote.xgb$class)
label.xgb <- ifelse(label.xgb == 1, 0, 1)


k.folds.xgb.xgb.smote<- function(k) {
  folds <- createFolds(thrombin.xgb$class, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    ## SMOTE is applied only to the training folds
    x <- SMOTE(class ~ ., thrombin.xgb[folds[[i]],], perc.over = 1000, perc.under=100)
    x$class <- as.numeric(x$class)
    x$class <- ifelse(x$class == 1, 0, 1)
    
    ## Model Building
    xgb <- xgboost(data = data.matrix(x[, -which(names(x) == "class") ], rownames.force = NA), 
                   label = x$class, 
                   eta = 0.1,
                   max_depth = 15, 
                   nround=25, 
                   subsample = 0.5,
                   colsample_bytree = 0.5,
                   seed = 1,
                   eval_metric = "auc",
                   objective = "binary:logistic",
                   nthread = 3)
    ## Prediction
    xgb.pred <- predict(xgb, 
                        newdata = data.matrix(thrombin.xgb[-folds[[i]], -which(names(thrombin.xgb) == "class") ], rownames.force = NA),
                        type = "response")
    
    ## Best cut-off value
    c_miss_error.xgb <- c()
    count <- 0
    for(c in xgb.pred){
      count <- count + 1
      c_miss_error.xgb[count] <- MISERR(thrombin.xgb$class[-folds[[i]]], xgb.pred, c = c)
    }
    
    c.xgb <- xgb.pred[c_miss_error.xgb == min(c_miss_error.xgb)]
    
    xgb.pred.values <- ifelse(xgb.pred >= min(c.xgb), 1, 0)
    
    
    ## Evaluation
    accuracies.xgb.xgb.smote <- c(accuracies.xgb.xgb.smote, 
                                  confusionMatrix(as.factor(xgb.pred.values), 
                                                  as.factor(thrombin.xgb$class[-folds[[i]]]))$byClass["Balanced Accuracy"])
    
    kappa.xgb.xgb.smote<- c(kappa.xgb.xgb.smote,
                            confusionMatrix(as.factor(xgb.pred.values), 
                                            as.factor(thrombin.xgb$class[-folds[[i]]]))$overall["Kappa"])
  }
  cbind(accuracies.xgb.xgb.smote, kappa.xgb.xgb.smote)
}

## 5-fold CV

ptm <- proc.time()
set.seed(567)
accuracies.xgb.xgb.smote <- c()
kappa.xgb.xgb.smote <- c()
eval.xgb.xgb.smote <- c()
eval.xgb.xgb.smote <- k.folds.xgb.xgb.smote(5)
proc.time()

eval.xgb.xgb.smote

## REPEATED 5-FOLD CV X50 TIMES

set.seed(567)
v <- c()
v <- replicate(50, k.folds.xgb.xgb.smote(5))
eval.xgb.xgb.smote <- c()

eval.xgb.xgb.smote <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.xgb.xgb.smote) <- c("Balanced Accuracy", "Kappa")


###################################################
##### XGBOOST, ELASTIC NET FEATURES, CLASS WEIGHTS

label.ela <- as.numeric(thrombin.ela$class)
label.ela <- ifelse(label.ela == 1, 0, 1)

## k-fold CV function is defined here
k.folds.xgb.ela.thrombin<- function(k) {
  folds <- createFolds(label.ela, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    
    ## Model Building
    xgb <- xgboost(data = data.matrix(thrombin.ela[folds[[i]], -which(names(thrombin.ela) == "class") ], rownames.force = NA), 
                   label = label.ela[folds[[i]]], 
                   eta = 0.1,
                   max_depth = 15, 
                   nround=25, 
                   subsample = 0.5,
                   colsample_bytree = 0.5,
                   seed = 1,
                   eval_metric = "auc",
                   objective = "binary:logistic",
                   nthread = 3,
                   scale_pos_weight = 44)
    
    ## Prediction
    xgb.pred <- predict(xgb, 
                        newdata = data.matrix(thrombin.ela[-folds[[i]], -which(names(thrombin.ela) == "class") ], rownames.force = NA),
                        type = "response")
    
    ## best cut-off value
    c_miss_error.xgb <- c()
    count <- 0
    for(c in xgb.pred){
      count <- count + 1
      c_miss_error.xgb[count] <- MISERR(label.ela[-folds[[i]]], xgb.pred, c = c)
    }
    
    c.xgb <- xgb.pred[c_miss_error.xgb == min(c_miss_error.xgb)]
    
    
    xgb.pred.values <- ifelse(xgb.pred >= min(c.xgb), 1, 0)
    
    ## Evaluation
    accuracies.xgb.ela.thrombin <- c(accuracies.xgb.ela.thrombin, 
                                     confusionMatrix(as.factor(xgb.pred.values), 
                                                     as.factor(label.ela[-folds[[i]]]))$byClass["Balanced Accuracy"])
    
    kappa.xgb.ela.thrombin<- c(kappa.xgb.ela.thrombin,
                               confusionMatrix(as.factor(xgb.pred.values), 
                                               as.factor(label.ela[-folds[[i]]]))$overall["Kappa"])
  }
  cbind(accuracies.xgb.ela.thrombin, kappa.xgb.ela.thrombin)
}

## 5-fold cross Validation
ptm <- proc.time()
set.seed(567)
accuracies.xgb.ela.thrombin <- c()
kappa.xgb.ela.thrombin <- c()
eval.xgb.ela.thrombin <- c()
eval.xgb.ela.thrombin <- k.folds.xgb.ela.thrombin(5)
proc.time() - ptm

eval.xgb.ela.thrombin


## REPEATED 5-FOLD CV X50 TIMES

set.seed(567)
v <- c()
v <- replicate(50, k.folds.xgb.ela.thrombin(5))
eval.xgb.ela.thrombin <- c()

eval.xgb.ela.thrombin <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.xgb.ela.thrombin) <- c("Balanced Accuracy", "Kappa")



###################################################
##### XGBOOST, LASSO FEATURES, CLASS WEIGHTS


label.lasso <- as.numeric(thrombin.lasso$class)
label.lasso <- ifelse(label.lasso == 1, 0, 1)

## k-fold CV function is defined here
k.folds.xgb.lasso.thrombin<- function(k) {
  folds <- createFolds(label.lasso, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    
    ## Model Building
    xgb <- xgboost(data = data.matrix(thrombin.lasso[folds[[i]], -which(names(thrombin.lasso) == "class") ], rownames.force = NA), 
                   label = label.lasso[folds[[i]]], 
                   eta = 0.1,
                   max_depth = 15, 
                   nround=25, 
                   subsample = 0.5,
                   colsample_bytree = 0.5,
                   seed = 1,
                   eval_metric = "auc",
                   objective = "binary:logistic",
                   nthread = 3,
                   scale_pos_weight = 44)
    
    ## Prediction
    xgb.pred <- predict(xgb, 
                        newdata = data.matrix(thrombin.lasso[-folds[[i]], -which(names(thrombin.lasso) == "class") ], rownames.force = NA),
                        type = "response")
    
    ## best cut-off value
    c_miss_error.xgb <- c()
    count <- 0
    for(c in xgb.pred){
      count <- count + 1
      c_miss_error.xgb[count] <- MISERR(label.lasso[-folds[[i]]], xgb.pred, c = c)
    }
    
    c.xgb <- xgb.pred[c_miss_error.xgb == min(c_miss_error.xgb)]
    
    
    xgb.pred.values <- ifelse(xgb.pred >= min(c.xgb), 1, 0)
    
    ## Evaluation
    accuracies.xgb.lasso.thrombin <- c(accuracies.xgb.lasso.thrombin, 
                                       confusionMatrix(as.factor(xgb.pred.values), 
                                                       as.factor(label.lasso[-folds[[i]]]))$byClass["Balanced Accuracy"])
    
    kappa.xgb.lasso.thrombin<- c(kappa.xgb.lasso.thrombin,
                                 confusionMatrix(as.factor(xgb.pred.values), 
                                                 as.factor(label.lasso[-folds[[i]]]))$overall["Kappa"])
  }
  cbind(accuracies.xgb.lasso.thrombin, kappa.xgb.lasso.thrombin)
}

## 5-fold CV

ptm <- proc.time()
set.seed(567)
threshold <- c()
accuracies.xgb.lasso.thrombin <- c()
kappa.xgb.lasso.thrombin <- c()
eval.xgb.lasso.thrombin <- c()
eval.xgb.lasso.thrombin <- k.folds.xgb.lasso.thrombin(5)
proc.time() - ptm

eval.xgb.lasso.thrombin

## REPEATED 5-FOLD SV X50 TIMES

set.seed(567)
v <- c()
v <- replicate(50, k.folds.xgb.lasso.thrombin(5))
eval.xgb.lasso.thrombin <- c()

eval.xgb.lasso.thrombin <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.xgb.lasso.thrombin) <- c("Balanced Accuracy", "Kappa")




###################################################
##### XGBOOST, XGB FEATURES, CLASS WEIGHTS


label.xgb <- as.numeric(thrombin.xgb$class)
label.xgb <- ifelse(label.xgb == 1, 0, 1)

## k-fold CV function is defined here
k.folds.xgb.xgb.thrombin<- function(k) {
  folds <- createFolds(label.xgb, k = k, list = TRUE, returnTrain = TRUE)
  for (i in 1:k) {
    
    ## Model Building
    xgb <- xgboost(data = data.matrix(thrombin.xgb[folds[[i]], -which(names(thrombin.xgb) == "class") ], rownames.force = NA), 
                   label = label.xgb[folds[[i]]], 
                   eta = 0.1,
                   max_depth = 15, 
                   nround=25, 
                   subsample = 0.5,
                   colsample_bytree = 0.5,
                   seed = 1,
                   eval_metric = "auc",
                   objective = "binary:logistic",
                   nthread = 3,
                   scale_pos_weight = 44)
    
    ## Prediction
    xgb.pred <- predict(xgb, 
                        newdata = data.matrix(thrombin.xgb[-folds[[i]], -which(names(thrombin.xgb) == "class") ], rownames.force = NA),
                        type = "response")
    
    ## Best cut-off value
    c_miss_error.xgb <- c()
    count <- 0
    for(c in xgb.pred){
      count <- count + 1
      c_miss_error.xgb[count] <- MISERR(label.xgb[-folds[[i]]], xgb.pred, c = c)
    }
    
    c.xgb <- xgb.pred[c_miss_error.xgb == min(c_miss_error.xgb)]
    
    
    xgb.pred.values <- ifelse(xgb.pred >= min(c.xgb), 1, 0)
    
    
    ## Evaluation
    accuracies.xgb.xgb.thrombin <- c(accuracies.xgb.xgb.thrombin, 
                                     confusionMatrix(as.factor(xgb.pred.values), 
                                                     as.factor(label.xgb[-folds[[i]]]))$byClass["Balanced Accuracy"])
    
    kappa.xgb.xgb.thrombin<- c(kappa.xgb.xgb.thrombin,
                               confusionMatrix(as.factor(xgb.pred.values), 
                                               as.factor(label.xgb[-folds[[i]]]))$overall["Kappa"])
  }
  cbind(accuracies.xgb.xgb.thrombin, kappa.xgb.xgb.thrombin)
}


## 5-fold CV
ptm <- proc.time()
set.seed(567)
accuracies.xgb.xgb.thrombin <- c()
kappa.xgb.xgb.thrombin <- c()
eval.xgb.xgb.thrombin <- c()
eval.xgb.xgb.thrombin <- k.folds.xgb.xgb.thrombin(5)
proc.time() - ptm

eval.xgb.xgb.thrombin

## REPEATED 5-FOLD CV X50 TIMES

set.seed(567)
v <- c()
v <- replicate(50, k.folds.xgb.xgb.thrombin(5))
eval.xgb.xgb.thrombin <- c()

eval.xgb.xgb.thrombin <- cbind(as.vector(v[, 1, 1:50]), as.vector(v[, 2, 1:50]))
colnames(eval.xgb.xgb.thrombin) <- c("Balanced Accuracy", "Kappa")


#############################################
## STORING EVALUATION METRICS IN A DATAFRAME

metrics<- rbind(eval.svm,
                eval.svm.xgb,
                eval.svm.xgb.smote,
                eval.svm.ela,
                eval.svm.ela.smote,
                eval.svm.lasso,
                eval.svm.lasso.smote,
                eval.glm,
                eval.glm.xgb.smote,
                eval.glm.lasso.smote,
                eval.glm.ela.smote,
                eval.xgb,
                eval.xgb.xgb.thrombin,
                eval.xgb.xgb.smote,
                eval.xgb.ela.thrombin,
                eval.xgb.ela.smote,
                eval.xgb.lasso.thrombin,
                eval.xgb.lasso.smote)

metrics.names <- c("svm", "svm.xgb", "svm.xgb.smote", "svm.ela", 
                   "svm.ela.smote", "svm.lasso", "svm.lasso.smote", 
                   "glm", "glm.xgb.smote", "glm.lasso.smote", 
                   "glm.ela.smote", "xgb", "xgb.xgb", 
                   "xgb.xgb.smote", "xgb.ela", "xgb.ela.smote", 
                   "xgb.lasso", "xgb.lasso.smote")

metrics <- as.data.frame(metrics)
metrics$class <- "1"

n <- 0
for(i in 1:4500){
  
  if(i%%250 == 1){
    n <- n + 1
  }
  metrics$class[i] <- metrics.names[n]
  
}

## Writing metrics in a csv. 
write.csv(metrics, "metrics.csv", row.names = FALSE)


## Common selected features are revealed here:
common.variables <- Reduce(intersect, list(selected.ela, 
                                           selected.lasso, 
                                           selected.xgb))

#  "V16405"  "V16799"  "V29154"  "V39500"  "V67657"  
#  "V79292"  "V79605"  "V79652"  "V79771"  "V106975"


#### PLOTTING THE METRICS ####

metrics$class <- as.factor(metrics$class)

## For a better representation, accuracy is multiplied with 100
metrics$`Balanced Accuracy` <- metrics$`Balanced Accuracy` * 100

## Models are ordered according to median values of balanced accuracy
metrics$class <- with(metrics, reorder(class, `Balanced Accuracy` , median))


par(mar = c(7, 4, 2, 2) +0.1)
boxplot( `Balanced Accuracy`~class, data=metrics, las = 2,
         main = "Balanced Accuracy Levels",
         ylab = "Balanced Accuracy (%)",
         frame = FALSE,
         col = c(rep("firebrick3", 3), 
                 rep("darkolivegreen",2), 
                 "cadetblue", 
                 "darkolivegreen",
                 "cadetblue",
                 rep("darkolivegreen",2),
                 "cadetblue",
                 rep("darkolivegreen",2),
                 "cadetblue",
                 "darkolivegreen",
                 "cadetblue",
                 "darkolivegreen",
                 "cadetblue"))
legend('bottomright', horiz = FALSE, 
       fill = c("firebrick3", "cadetblue", "darkolivegreen"), 
       legend = c("Original", "Selected, Class Weight", "Selected, SMOTE"), 
       bty = 'n')


## Models are ordered according to median values of kappa
metrics$class <- with(metrics, reorder(class, Kappa, median))

par(mar = c(7, 4, 2, 2) +0.1)
boxplot(Kappa~class, data=metrics, las = 2,
        main = "Cohen's Kappa Values",
        ylab = "Kappa",
        frame = FALSE,
        col = c("darkolivegreen",
                rep("firebrick3", 3), 
                rep("darkolivegreen",5),
                "cadetblue",
                rep("darkolivegreen",2),
                "cadetblue",
                "darkolivegreen",
                rep("cadetblue",4)),
        cex.axis = 1)
legend('bottom', horiz = FALSE, 
       fill = c("firebrick3", "cadetblue", "darkolivegreen"), 
       legend = c("Original", "Selected, Class Weight", "Selected, SMOTE"), 
       bty = 'n')


### Checking the normality for Balanced Accuracy by Shapiro-Wilk test

listids <- list()
for (ids in unique(metrics$class)){
  subdf <- subset(x=metrics, subset=class==ids)
  # apply the rest of your analysis there using subdf, for instance 
  listids[[ids]] <- shapiro.test(subdf$`Balanced Accuracy`)
}

### Checking the normality for Kappa by Shapiro-Wilk test

listids <- list()
for (ids in unique(metrics$class)){
  subdf <- subset(x=metrics, subset=class==ids)
  # apply the rest of your analysis there using subdf, for instance 
  listids[[ids]] <- shapiro.test(subdf$Kappa)
}

## Kruskal Wallis Tests for significance of difference

kruskal.test(metrics$`Balanced Accuracy`~ metrics$class)
kruskal.test(metrics$Kappa~ metrics$class)



#######################################################
################# THE PREDICTION ######################
#######################################################


## SVM MODEL BUILDING WITH LASSO FEATURES

model.svm.lasso <- svm(thrombin.lasso$class~.,
                       data = thrombin.lasso[, -which(names(thrombin.lasso) == "class")],
                       cost = 4,
                       gamma = 0.01,
                       type = "C",
                       kernel = "linear",
                       class.weights = c("0" = 44, "1" = 1))

## SVM MODEL BUILDING WITH ELASTIC NET FEATURES

model.svm.ela <- svm(thrombin.ela$class~.,
                     data = thrombin.ela[, -which(names(thrombin.ela) == "class")],
                     cost = 4,
                     gamma = 0.01,
                     type = "C",
                     kernel = "linear",
                     class.weights = c("0" = 44, "1" = 1))
  

## XGB MODEL BUILDING WITH LASSO FEATURES

# XGB ACCEPTS NUMERIC VALUES FOR THE RESPONSE
label.lasso.model <- thrombin.lasso$class
label.lasso.model <- as.numeric(label.lasso.model)
label.lasso.model <- ifelse(label.lasso.model == 1, 0, 1)

model.xgb.lasso <- xgb <- xgboost(data = data.matrix(thrombin.lasso[, -which(names(thrombin.lasso) == "class") ], rownames.force = NA), 
                                  label = label.lasso.model, 
                                  eta = 0.1,
                                  max_depth = 15, 
                                  nround=25, 
                                  subsample = 0.5,
                                  colsample_bytree = 0.5,
                                  seed = 1,
                                  eval_metric = "auc",
                                  objective = "binary:logistic",
                                  nthread = 3,
                                  scale_pos_weight = 44)


#### THE PREDICTION WITH THREE MODELS


svm.ela.pred <- predict(model.svm.ela, 
                        newdata = test.ela,
                        type = "class")


svm.lasso.pred <- predict(model.svm.lasso, 
                        newdata = test.lasso,
                        type = "class")


xgb.pred <- predict(model.xgb.lasso, 
                    newdata = data.matrix(test.lasso, rownames.force = NA),
                    type = "class")


## Threshold is decided along with the histogram of the probabilities
## Since it is not normally distributed, median and quantiles are used
## Probabilities that are out of 80% quantile are classified as Active

hist(xgb.pred)
c.xgb <- quantile(xgb.pred, probs = seq(0, 1, 0.10))["80%"]

xgb.lasso.pred <- ifelse(xgb.pred >= c.xgb, 1, 0)



## Active - Inactive ones are checked with tables
table(svm.lasso.pred)
table(svm.ela.pred)
table(xgb.lasso.pred)

summary(metrics[metrics$class == "svm.ela", 1])
summary(metrics[metrics$class == "svm.lasso", 1])
summary(metrics[metrics$class == "xgb.lasso", 1])

## As can be seen in the tables, there are many active ones are listed
## And since we know the training dataset consists of many inactive ones
## We expect that the test data set is also imbalanced. 

### That's why majority vote will be the final decision on the prediction
### All the predictions that are classified as active in each set 
### is defined as an active in the test set. 

predictions <- cbind(svm.lasso.pred, svm.ela.pred, xgb.lasso.pred)

predictions <- ifelse(predictions == 1, 0, 1)


## Creating the Majority Votes
majority <- rowSums(predictions)

majority <- ifelse(majority == 3, 1, 0)

## Checking the predictions
table(majority)

## Saving the prediction column to a txt file. 
majority.letter <- ifelse(majority == 1, "A", "I")

## Saving the file

save(majority.letter, file = "Bayrak-Cartuyvels-Predictions.RData")

