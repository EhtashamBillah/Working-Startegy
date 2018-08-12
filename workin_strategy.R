
#setting up the strategy
# install.packages("purrr")


# install.packages("class")
require(class) # for knn
# install.packages("e1071")
require(e1071) #svm,kernel svm,bayes
# install.packages("rpart")
require(rpart) # for decision tree
# install.packages("randomForest")
require(randomForest) # for random forest
# install.packages("caret")
require(caret) # boss of all
# install.packages("bartMachine")
require(bartMachine) #for bayesian regression trees
# install.packages("C50")
require(C50)
# install.packages("fastAdaboost")
require(fastAdaboost)
# install.packages("xgboost")
require(xgboost)
# install.packages("adabag")
require(adabag)
# install.packages("caretEnsemble")
require(caretEnsemble) # for stacking models
# install.packages("kolmim")
require(kolmim) # for Kolmogorov-Smirnov test for classification
# install.packages("plotly")
require(plotly)
# install.packages("plot3D")
require(plot3D)
# install.packages("learningCurve")
require(learningCurve)
# install.packages("doParallel")
require(doParallel) 
# install.packages("doSNOW")
require(doSNOW) # for parallel computing

# distributing computations among # of coreS in CPU
getDoParWorkers()
getDoParRegistered()
getDoParName()
getDoParVersion()
cl <- makeCluster(spec=4, type="SOCK")
registerDoSNOW(cl) 
stopCluster(cl)



income<-round(runif(100,min=1000,max=5000),0)
no_of_wife <-round(rnorm(100,mean=25,sd=8),0)
no_of_child<-round(rnorm(100,mean=100,sd=45),0)
alive <- rbernoulli(100, p = 0.6)
dataset<- data.frame(income,no_of_wife,no_of_child,alive)


#DATA PREPROCESSING
# before labeling, execute the below code first,then inside the labels function write as it is
make.names(dataset$alive)
dataset$alive <- factor(dataset$alive,levels=c("TRUE","FALSE"),labels=c("TRUE.","FALSE."))
levels(dataset$alive) # should have to be similar to make.names(dataset$alive). otherwise will create problem in ensemble ML

# checking multicollinerity
cor(dataset[,1:3])
plot(dataset[,1:3])

#checking for normality and outliers
apply(dataset,2,skewness)

#####################################################
# data splitting
require(caTools)
split <- sample.split(dataset$alive,SplitRatio = 0.75)
training_set <- subset(dataset,split==T)
test_set <- subset(dataset,split==F)

# feature scalling
training_set[,-4] <- scale(training_set[,-4])
test_set[,-4] <- scale(test_set[,-4])


###################################
#K_FOLD CROSS VALIDATION
#WILL CROSS VALIDATE
#LOGISTIC,KNN,SVM,KERNEL SVM,DECISION TREE,RANDOM FOREST,NAIVE BAYES,XGBOOST,ADABOOST
set.seed(12345)
# folds<- createMultiFolds(training_set$alive,k=10,times=10) [ NOT IN KUHN'S SLIDE]
Control <- trainControl(method="repeatedcv",number=10,repeats = 10,classProbs = T,summaryFunction = twoClassSummary)# for CALCULATING ROC classProbS=T IS REQUIRED


# use tuneLength or tunegrid parameter to find the optimum performance of each model
#Use getModelInfo() to get a list of tuning parameters for each model or see http://topepo.github.io/caret/available-models.html.
set.seed(12345)
model_svm_cross <- train(form=alive~.,data=training_set,
                         trControl=Control,tuneLength=5 ,method="svmLinear2",metric="ROC") # library e1071.can also use metric="ROC",tuneGrid <- expand.grid(cost(0.01,0.1,0,25))

set.seed(12345)
model_ksvm_cross <- train(form=alive~.,data=training_set,
                          trControl=Control,tuneLength=5,method="svmRadial",metric="Accuracy")  # library kernlab.can also use metric="ROC"

set.seed(12345)
model_bayes2_cross <- train(form=alive~.,data=training_set,
                            trControl=Control,tuneLength=5,method="naive_bayes",metric="Accuracy") #library naivebayes.can also use metric="ROC"


set.seed(12345)
model_knn_cross <- train(form=alive~.,data=training_set,ntree=500,
                        tuneLength=5,method="kknn",metric="Accuracy",trControl=Control)
#check for overfitting
#install.packages("learningCurve")
require(learningCurve)


# variable importance
varImp(model_svm_cross)
varImp(model_ksvm_cross)
varImp(model_bayes2_cross)
varImp(model_knn_cross)

#ploting the models
plot(model_svm_cross)
plot(model_ksvm_cross)
plot(model_bayes2_cross)
plot(model_knn_cross)
plotly(training_set)
hist3D(model_knn_cross)

########################################
#storing the results
results <- resamples(list(Bayes=model_bayes2_cross,KSVM=model_ksvm_cross,svm=model_svm_cross,knn=model_knn_cross))
summary(results)
dotplot(results) 
bwplot(results)

#from here we get our best model


#######################################
# model evaluation and diagonistics(for best models)
########################################

# Goodness of fit
    # a) LRT (FOR REGRESSION)
    # B) psedu R2 (FOR LOGISTIC REGRESSION)
# Statistical Tests for Individual Predictors
    # Wald Test (FOR REGRESSION)
    # Variable Importance

# STATISTICAL TEST FOR CLASSIFICATION
    # Kolmogorov-Smirnov TEST

#PERFORMANCE MEASURE BY Kolmogorov-Smirnov TEST IN R
require(kolmim)
ks.test.imp(x=training_set$income,"pexp") # FIGURE HOW IT WORKS


#######################################
# validation the predicted values

# 1. confusion matrix
confusionMatrix(y_hat,test_set$alive)
## kappa statistic
# kappa = (O-E)/(1-E) #where O is the observed accuracy and E is the expected accuracy under chance agreement

# 2.ROC curve ( FOR 2 CLASS MODEL)
# The receiving operating characteristic is a measure of classifier performance. Using the proportion of positive data points that are correctly considered as positive and the proportion of negative data points that are mistakenly considered as positive(i.e false positive), we generate a graphic that shows the trade off between the rate at which you can correctly predict something with the rate of incorrectly predicting something. Ultimately, we're concerned about the area under the ROC curve, or AUROC. That metric ranges from 0.50 to 1.00, and values above 0.80 indicate that the model does a good job in discriminating between the two categories which comprise our target variable. Bear in mind that ROC curves can examine both target-x-predictor pairings and target-x-model performance.
# install.packages("pROC")
require(pROC)
# Compute AUC for predicting y with individual independent variable
f1 <- roc(alive~income,data = training_set) #shows the trade off between the rate at which you can correctly predict something with the rate of incorrectly predicting something
plot(f1,col="red")
f2 <- roc(alive~no_of_wife,data = training_set)
plot(f2,col="blue")
f3 <- roc(alive~no_of_child,data = training_set)
plot(f3,col="green")
# auc() gives numeric value of area under curve which we can also get from f1,f2,f3 as well
auc(f1) # Ultimately, we're concerned about the area under the ROC curve, or AUROC. That metric ranges from 0.50 to 1.00, and values above 0.80 indicate that the model does a good job in discriminating between the two categories which comprise our target variable
auc(f2)
auc(f3)
#confidence interval of roc curve.
ci(f1)
ci(f2)
ci(f3)

#Compute AUC for predicting Class with the model
#install.packages("ROCR")
require(ROCR)

y_hat <- predict(model_ksvm_cross,newdata=test_set[,-4],type="response")
pred <- prediction(predictions = as.numeric(y_hat),labels=test_set$alive)
perform <- performance(pred,measure = "tpr",x.measure="fpr")
plot(perform)
auc <- performance(pred,measure = "auc")
auc
auc@y.values[[1]]


###############################################
#ensemble modeling (ensembling the best models)
##############################################
# there are 3 most popular methods for combining the predictions from different models
 # 1.Bagging : Building multiple models (typically of the same type) from different subsamples of the training dataset.
 # 2.Boosting : Building multiple models (typically of the same type) each of which learns to fix the prediction errors of a prior model in the chain.
 # 3.Stacking : Building multiple models (typically of differing types) and supervisor model that learns how to best combine the predictions of the primary models.

####################
# 1. BOOSTING (C50,XGBOOST,ADABOOST,STOCHASTIC GRADIENT BOOSTING ETC)
###################

set.seed(12345)
model_c50_cross <- train(form=alive~.,data=training_set,tuneLength=5,
                         trControl=Control,method="C5.0",metric="Accuracy")


set.seed(12345)
model_gbm_cross <- train(form=alive~., data=training_set,tuneLength=5,
                         trControl=Control,method="gbm",metric="Accuracy")


set.seed(12345)
model_xgb_cross <- train(form=alive~.,data=training_set,tuneLength=5,
                         trControl = Control,method="xgbTree",metric="Accuracy")


set.seed(12345)
model_ada_cross <- train(form=alive~.,data=training_set,tuneLength=5,
                         trControl=Control,method="adaboost",metric="Accuracy")



set.seed(12345)
# ran for 3 hours..stopped it
# model_adaM1_cross <- train(form=alive~.,data=training_set,tuneLength=5,
                         # trControl=Control,method="AdaBoost.M1")

#check for overfitting


# variable importance
varImp(model_c50_cross) # did not worked
varImp(model_gbm_cross) # did not worked
varImp(model_xgb_cross)
varImp(model_ada_cross)

#ploting the models
plot(model_c50_cross)
plot(model_gbm_cross)
plot(model_xgb_cross)
plot(model_ada_cross)


########################################
#storing the results
boosting_results <- resamples(list(C50=model_c50_cross,GBM=model_gbm_cross,XGB=model_xgb_cross,ADABOOST=model_ada_cross))
summary(boosting_results)
dotplot(boosting_results) 
bwplot(boosting_results)


#####################################
# 2. bagging (treebag,random forest)
####################################

set.seed(12345)
model_treebag_cross <- train(form=alive~.,data=training_set,tuneLength=5,
                             trControl=Control,method="treebag",metric="Accuracy")
set.seed(12345)
model_rf_cross <- train(form=alive~.,data=training_set,tuneLength=5,
                             trControl=Control,method="rf",metric="Accuracy")

#check for overfitting


# variable importance
varImp(model_treebag_cross) 
varImp(model_rf_cross) 

#ploting the models
plot(model_treebag_cross) # did not worked.
plot(model_rf_cross)



#########################################
#storing the results
baggting_results <- resamples(list(TREEBAG=model_treebag_cross,RF=model_rf_cross))
summary(baggting_results)
dotplot(baggting_results) 
bwplot(baggting_results)



#####################################
# 2. stacking 
####################################

# creating submodels
control <- trainControl(method = "repeatedcv",number=10,repeats = 10,index=folds,savePredictions = T,classProbs = T)
algorithmList <- c("lda","kknn","naive_bayes","svmRadial")
set.seed(12345)
models <- caretList(alive~.,data=training_set,trControl = control,methodList = algorithmList)
class(models) # Create a list of several train models from the caret package Build a list of train objects suitable for ensembling using the caretEnsemble function

#storing results
comb_results <- resamples(models)     
summary(comb_results)
dotplot(comb_results) 
bwplot(comb_results)
#correlation between results
modelCor(comb_results)
splom(comb_results)


#stack using svmRadial
stackcontrol <- trainControl(method = "repeatedcv",number=10,repeats = 10,savePredictions = T,classProbs = T)
set.seed(12345)
stack.svmRadial <- caretStack(models,method="svmRadial",trControl=stackcontrol,metric="Accuracy") # THIS IS ANOTHER MODEL
class(stack.svmRadial) # Combine several predictive models via stacking
#stack using RF
stackcontrol <- trainControl(method = "repeatedcv",number=10,repeats = 10,savePredictions = T,classProbs = T)
set.seed(12345)
stack.rf <- caretStack(models,method="rf",trControl=stackcontrol,metric="Accuracy") # THIS IS ANOTHER MODEL
#stack using glm
stackcontrol <- trainControl(method = "repeatedcv",number=10,repeats = 10,savePredictions = T,classProbs = T)
set.seed(12345)
stack.glm <- caretStack(models,method="glm",trControl=stackcontrol,metric="Accuracy") # THIS IS ANOTHER MODEL

#check for overfitting

# variable importance
varImp(stack.svmRadial) # did not worked
varImp(stack.rf) # did not worked
varImp(stack.glm) # did not worked

#ploting the models
plot(stack.svmRadial)
plot(stack.rf)
plot(stack.glm) # did not worked

# caret ensembel
caretEnsemble()


# predict y using differnt stacked model
y_hat1 <- predict(stack.svmRadial,newdata=test_set[,-4])
y_hat2 <- predict(stack.rf,newdata=test_set[,-4])
y_hat3 <- predict(stack.glm,newdata=test_set[,-4])

confusionMatrix(y_hat1,test_set$alive) # RESULT IS SAME for y_hat1,y_hat2,y_hat3 BUT MUCH LESS THEN STACKED MODELS
confusionMatrix(y_hat2,test_set$alive) # RESULT IS SAME for y_hat1,y_hat2,y_hat3 BUT MUCH LESS THEN STACKED MODELS
confusionMatrix(y_hat3,test_set$alive) # RESULT IS SAME for y_hat1,y_hat2,y_hat3 BUT MUCH LESS THEN STACKED MODELS


