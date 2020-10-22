# Understanding the data
library(caret)
raw <- read.csv("bank.csv",header=TRUE)
str(raw)
summary(raw)

# Check unique values by each variables
unique(raw$age)
unique(raw$job)
unique(raw$marital)
unique(raw$education)
unique(raw$default)
unique(raw$housing)
unique(raw$loan)
unique(raw$contact)
unique(raw$month)
unique(raw$day_of_week)
unique(raw$duration)
unique(raw$campaign)
unique(raw$previous)
unique(raw$poutcome)
unique(raw$emp.var.rate)
unique(raw$cons.price.idx)
unique(raw$cons.conf.idx)
unique(raw$euribor3m)
unique(raw$nr.employed)
unique(raw$target)

## Since there are 'unknown' values in the data, change it to NA and delete
raw[raw=='unknown'] <- NA
sum(is.na(raw))
raw <- na.omit(raw)
sum(is.na(raw))

## Since the data are in Numeric/Categoric values, separate data based on the type. 
# 1. Numeric Variables : age, duration, campaign, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed
# Normalization
raw$age <- scale(raw$age)
raw$duration <- scale(raw$duration)
raw$campaign <- scale(raw$campaign)
raw$previous <- scale(raw$previous)
raw$emp.var.rate <- scale(raw$emp.var.rate)
raw$cons.price.idx <- scale(raw$cons.price.idx)
raw$cons.conf.idx <- scale(raw$cons.conf.idx)
raw$euribor3m <- scale(raw$euribor3m)
raw$nr.employed <- scale(raw$nr.employed)

# Histogram (with numeric values that are normalized)
par(mfrow=c(3,3),mar=c(5,4,4,2))
hist(raw$age,main='Age',xlab='Age',col='red')
hist(raw$duration,main='Duration', xlab='Last Contact Duration',col='orange')
hist(raw$campaign,main='Campaign',xlab = 'Last Contact Frequency',col='yellow')
hist(raw$previous,main='Previous marketing',xlab='Previous marketing contact frequency',col='green')
hist(raw$emp.var.rate,main='Employment Variation Rate',xlab='emp.var.rate',col='blue')
hist(raw$cons.price.idx,main='Consumer price index',xlab='cons.price.idx',col='navy')
hist(raw$cons.conf.idx,main='Consumer confidence index',xlab='cons.conf.idx',col='purple')
hist(raw$euribor3m,main='Euribo Rate',xlab='Euribo rate (3months)',col='gray')
hist(raw$nr.employed,main='Employees',xlab='Number of Employees',col='black')

# 2. Categorized variables : job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome, factor
## 1) Barplot based on proportional
par(mfrow=c(3,3),mar=c(5,4,4,2))
barplot(prop.table(table(raw$job)),main='Job ratio')
barplot(prop.table(table(raw$marital)),main='Marriage status')
barplot(prop.table(table(raw$education)),main='Education level')
barplot(prop.table(table(raw$default)),main='Bankruptcy status')
barplot(prop.table(table(raw$housing)),main='Housing loan status')
barplot(prop.table(table(raw$loan)),main='Loan status')
barplot(prop.table(table(raw$contact)),main='Contact method')
barplot(prop.table(table(raw$month)),main='Last contact month')
barplot(prop.table(table(raw$day_of_week)),main='Last contact day')

## 2) Pie chart based on proportional
par(mfrow=c(3,3),mar=c(1,1,1,1))
pie(prop.table(table(raw$job)),main='Job ratio')
pie(prop.table(table(raw$marital)),main='Marriage status')
pie(prop.table(table(raw$education)),main='Education level')
pie(prop.table(table(raw$default)),main='Bankruptcy status')
pie(prop.table(table(raw$housing)),main='Housing loan status')
pie(prop.table(table(raw$loan)),main='Loan status')
pie(prop.table(table(raw$contact)),main='Contact method')
pie(prop.table(table(raw$month)),main='Last contact month')
pie(prop.table(table(raw$day_of_week)),main='Last contact day')


## Since there are 9 principle components, let's use PCA to reduce those.
num_feature <- c('age','duration','campaign','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed') # Variables with numerical features
tar <- raw[,'target']
num_data <- raw[,num_feature] # Only continuous numeric features

pca_num <- prcomp(num_data)
plot(pca_num,type='l',main='Principle Component Analysis (PCA)')

summary(pca_num)
## It seems to cut it down to 5 principle components, but it is unnecessary to cut 9 to 6 components. It won't show any big difference. Therefore, PCA method is unnecessary in this case.

## But let's just try after reducing it into 2 and 3 principle components to check if the results are better than before.
pca_matrix <- pca_num$rotation
pca_data <- as.matrix(num_data) %*% pca_matrix
dim(pca_data)
reduced_data <- data.frame(cbind(pca_data[,1:3],tar))
reduced_data$tar <- as.factor(reduced_data$tar)

library(ggplot2)
ggplot(data=reduced_data, aes(x=PC1,y=PC2))+
  geom_point(aes(color=tar,shape=tar))+
  xlab('PC1')+
  ylab('PC2')+
  ggtitle('PCA : 2 principle components')

library(scatterplot3d)
shapes=c(16,17)
shapes <- shapes[as.numeric(reduced_data$tar)]
scatterplot3d(reduced_data[,1:3],color=reduced_data[,'tar'],pch=shapes,angle=30)

### PCA method is not useful in this case.

# Test/Train test
set.seed(5)
new <- raw
datatotal <- sort(sample(nrow(new),nrow(new)*0.7))
train <- new[datatotal,]
test <- new[-datatotal,]

# Machine Learning Method 
## 1. Logistic Regression
ctrl <- trainControl(method="repeatedcv",repeats=5)
logit_fit <- train(target~.,
                   data=train,
                   method="glm",
                   trControl=ctrl,
                   metric="Accuracy")
logit_pred <- predict(logit_fit,newdata=test)

# ML : Logistic Regression
ctrl <- trainControl(method='repeatedcv',repeats=5)
logit_fit <- train(target~.,
                   data=train,
                   method='glm',
                   trControl=ctrl,
                   metric='Accuracy')
logit_fit
logit_pred <- predict(logit_fit,newdata=test)
confusionMatrix(logit_pred,test$target)

# ML : Boosted Logistic Regression
ctrl <- trainControl(method='repeatedcv',repeats=5)
logit_boost_fit <- train(target~.,
                         data=train,
                         method='LogitBoost',
                         trControl=ctrl,
                         metric='Accuracy')
plot(logit_boost_fit)
## ???????????? ????????? ????????????????????????????????? ??????!

logit_boost_pred <- predict(logit_boost_fit,newdata=test)
confusionMatrix(logit_boost_pred,test$target)

# ML : Logistic Model Tree
########################
ctrl <- trainControl(method='repeatedcv',repeats=5)
logit_tree_fit <- train(target~.,
                        data=train,
                        method="LMT",
                        trControl=ctrl,
                        metric='Accuracy')

plot(logit_tree_fit)

logit_tree_pred <- predict(logit_boost_fit,newdata=test)
confusionMatrix(logit_tree_pred,test$target)
###################

# ML : penalized logistic regression
logit_plr_fit <- train(target~.,
                       data=train,
                       method="plr",
                       trControl=ctrl,
                       metric='Accuracy')
plot(logit_plr_fit)
logit_plr_pred <- predict(logit_plr_fit,newdata=test)
confusionMatrix(logit_plr_pred,test$target)

# ML : regularized logistic
########################
ctrl <- trainControl(method='repeatedcv',repeats=5)
logit_reg_fit <- train(target~.,
                       data=train,
                       method="regLogistic",
                       trControl=ctrl,
                       metric='Accuracy')
plot(logit_reg_fit)
logit_reg_pred <- predict(logit_reg_fit,newdata=test)
confusionMatrix(logit_reg_pred,test$target)

# ML : Naive Bayes
ctrl <- trainControl(method='repeatedcv',repeats=5)
nb_fit <- train(target~.,
                data=train,
                method="naive_bayes",
                trControl=ctrl,
                metric="Accuracy")
plot(nb_fit)

nb_pred <- predict(nb_fit,newdata=test)
confusionMatrix(nb_pred,test$target)

# ML : RandomForest
ctrl <- trainControl(method='repeatedcv',repeats=5)
rf_fit <- train(target~.,
                data=train,
                method='rf',
                trControl=ctrl,
                metric='Accuracy')
rf_fit

rf_pred <- predict(rf_fit,newdata=test)
confusionMatrix(rf_pred,test$target)

# ML : Support Vector Machine
ctrl <- trainControl(method='repeatedcv',repeats=5)
svm_Linear_fit <- train(target~.,
                        data=train,
                        method='svmLinear',
                        trControl=ctrl,
                        metric='Accuracy')
svm_Linear_fit <- predict(svm_Linear_fit,newdata=test)
confusionMatrix(svm_Linear_fit,test$target)

# ML : Kernel Support Vector Machine
ctrl <- trainControl(method='repeatedcv',repeats=5)
svm_poly_fit <- train(target~.,
                      data=train,
                      method='svm_poly_fit',
                      trControl=ctrl,
                      metric='Accuracy')
plot(svm_poly_fit)
svm_poly_pred <- predict(svm_poly_pred,newdata=test)
confusionMatrix(svm_poly_pred,test$target)

# ????????? ??????
## ????????? ????????? ??????????!!?!?!?!?!??

# RandomForest!~!! (test data > train data) 







