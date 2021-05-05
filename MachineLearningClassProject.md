---
title: "MachineLearningClassProject"
author: "Gabe Carvalho"
date: "April, 2021"
output:
  html_document:
    keep_md: yes

---



## Synopsis

This analysis will explore the ability of machine learning to use data from wearable devices to determine what activities wearers were performing. Information on the dataset can be viewed here <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>.  



```r
trainData<-read.csv("pml-training.csv")
testData<-read.csv("pml-testing.csv")
```

## Data Preparation

Before the modle is run the appropriate predictors must be selected. There being 160 variables most will need to be eliminated for the model to run. First some variables of junk and redundant data are removed. Then a random forest is run to determine the which group of variables will be the best predictors. These will then be stored in rfe_lm_profile and used for the model.


```r
##Remove columns of time stamps and user names (these will cause over prediction in the training data)
trainData <- trainData[, -c(1:7)]
testData <- testData[, -c(1:7)]

##replace NA with 0
trainData <-trainData[ , colSums(is.na(trainData)) == 0]

##Remove near zero variance columns to speed up predictor selection
nzv <-nearZeroVar(trainData)
trainData <-trainData[, -nzv]

##Random Forest predictor slection
ctrl_param <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 2,
                   number=3,
                   verbose = FALSE,
                   allowParallel = TRUE,
                   returnResamp = "all")
rfe_lm_profile <- rfe(classe~., trainData,
                 sizes = c(4,6,8,10),
                 mtry=6,
                 rfeControl = ctrl_param,
                 na.action = na.pass)
##summary(trainData)
rfe_lm_profile
```

```
## 
## Recursive feature selection
## 
## Outer resampling method: Cross-Validated (3 fold, repeated 2 times) 
## 
## Resampling performance over subset size:
## 
##  Variables Accuracy  Kappa AccuracySD  KappaSD Selected
##          4   0.9395 0.9235  0.0033790 0.004286         
##          6   0.9728 0.9656  0.0041962 0.005309         
##          8   0.9825 0.9779  0.0009054 0.001145         
##         10   0.9850 0.9811  0.0016693 0.002111         
##         52   0.9943 0.9928  0.0009305 0.001178        *
## 
## The top 5 variables (out of 52):
##    roll_belt, yaw_belt, magnet_dumbbell_z, pitch_belt, magnet_dumbbell_y
```

```r
predictors(rfe_lm_profile)
```

```
##  [1] "roll_belt"            "yaw_belt"             "magnet_dumbbell_z"   
##  [4] "pitch_belt"           "magnet_dumbbell_y"    "pitch_forearm"       
##  [7] "accel_dumbbell_y"     "magnet_forearm_z"     "roll_forearm"        
## [10] "roll_dumbbell"        "accel_dumbbell_z"     "roll_arm"            
## [13] "magnet_dumbbell_x"    "magnet_belt_z"        "gyros_belt_z"        
## [16] "magnet_belt_y"        "yaw_arm"              "magnet_belt_x"       
## [19] "accel_forearm_x"      "gyros_arm_y"          "magnet_forearm_y"    
## [22] "yaw_dumbbell"         "gyros_forearm_y"      "gyros_dumbbell_y"    
## [25] "accel_forearm_z"      "accel_belt_z"         "total_accel_dumbbell"
## [28] "magnet_arm_z"         "gyros_arm_x"          "accel_forearm_y"     
## [31] "gyros_dumbbell_x"     "accel_dumbbell_x"     "yaw_forearm"         
## [34] "accel_arm_y"          "gyros_forearm_z"      "pitch_arm"           
## [37] "total_accel_forearm"  "gyros_dumbbell_z"     "magnet_forearm_x"    
## [40] "accel_arm_z"          "total_accel_arm"      "accel_arm_x"         
## [43] "magnet_arm_x"         "magnet_arm_y"         "gyros_belt_x"        
## [46] "gyros_forearm_x"      "accel_belt_x"         "pitch_dumbbell"      
## [49] "total_accel_belt"     "gyros_belt_y"         "gyros_arm_z"         
## [52] "accel_belt_y"
```

## Modeling

For this analysis the model will split the the training set into 10 folds, using 9 of the folds to train the model and the 10th to test it. This process is repeated 10 times with each fold used as the reserved test set 1 time.


```r
y<-as.matrix(select(trainData,"classe"))
x<-select(trainData,c(predictors(rfe_lm_profile)))
train_control<- trainControl(method="cv", number=10, savePredictions = TRUE, allowParallel = TRUE)
modelFit<-train(x,y,trControl=train_control,  na.action  = na.pass)

# control <- trainControl(method='repeatedcv', 
#                         number=10, 
#                         repeats=3,
#                         allowParallel = TRUE)
# mtry <- sqrt(ncol(x))
# tunegrid <- expand.grid(.mtry=mtry)
# modelFit<<- train(x,y, 
#                       method='rf', 
#                       na.action  = na.omit,
#                       metric='Accuracy', 
#                       tuneGrid=tunegrid, 
#                       trControl=control)


## Stop parallel processing
stopCluster(cluster)
registerDoSEQ()
```
## Predicting

Here we see the summary of the model's fit and then try to predict the test data results. Accuracy for 27 variables is highest at 99.5%, so the 27 variable model will be used to predict the test data.


```r
summary(modelFit$pred)
```

```
##  pred      obs          rowIndex          mtry      Resample        
##  A:16797   A:16740   Min.   :    1   Min.   : 2   Length:58866      
##  B:11371   B:11391   1st Qu.: 4906   1st Qu.: 2   Class :character  
##  C:10334   C:10266   Median : 9812   Median :27   Mode  :character  
##  D: 9573   D: 9648   Mean   : 9812   Mean   :27                     
##  E:10791   E:10821   3rd Qu.:14717   3rd Qu.:52                     
##                      Max.   :19622   Max.   :52
```

```r
modelFit
```

```
## Random Forest 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17660, 17660, 17660, 17661, 17658, 17660, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9956684  0.9945207
##   27    0.9953116  0.9940693
##   52    0.9901133  0.9874922
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

```r
##predict(modelFit, newdata=as.matrix(select(testData,c(predictors(rfe_lm_profile)))))
```
