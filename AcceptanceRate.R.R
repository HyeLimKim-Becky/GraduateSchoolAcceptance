# Load data
rawdata <- read.csv("university.csv",header=TRUE)
str(rawdata)
summary(rawdata)
sum(is.na(rawdata))

# Understanding the data
# Check unique values by variables
unique(rawdata$GRE.Score)
unique(rawdata$TOEFL.Score)
unique(rawdata$University.Rating)
unique(rawdata$SOP)
unique(rawdata$LOR)
unique(rawdata$CGPA)
unique(rawdata$Research)
unique(rawdata$Chance.of.Admit)
Uni_Rating_Table <- table(rawdata$University.Rating)
Research_Table <- table(rawdata$Research)

# Histogram
par(mfrow=c(3,3),mar=c(5,4,4,1))
hist(rawdata$GRE.Score,main='GRE',xlab='GRE score',col='red')
hist(rawdata$TOEFL.Score,main='TOEFL',xlab='Toefl score',col='orange')
hist(rawdata$University.Rating,main='University Rating',xlab='University rating',col='black')
hist(rawdata$SOP,main='SOP',xlab='sop',col='yellow')
hist(rawdata$LOR,main='LOR',xlab='lor',col='green')
hist(rawdata$CGPA,main='CGPA',xlab='cgpa',col='blue')
hist(rawdata$Research,main='Research Experience',xlab='research',col='navy')
hist(rawdata$Chance.of.Admit,main='Chance of Admit',xlab='Admission chance',col='purple')

# Boxplot
## Double checking with boxplot to check outlier
par(mfrow=c(3,3),mar=c(2,4,4,2))
boxplot(rawdata$GRE.Score,main='GRE',xlab='GRE score',col='red')
boxplot(rawdata$TOEFL.Score,main='TOEFL',xlab='Toefl score',col='orange')
boxplot(rawdata$University.Rating,main='University Rating',xlab='University rating',col='black')
boxplot(rawdata$SOP,main='SOP',xlab='sop',col='yellow')
boxplot(rawdata$LOR,main='LOR',xlab='lor',col='green')
boxplot(rawdata$CGPA,main='CGPA',xlab='cgpa',col='blue')
boxplot(rawdata$Research,main='Research Experience',xlab='research',col='navy')
boxplot(rawdata$Chance.of.Admit,main='Chance of Admit',xlab='Admission chance',col='purple')

# Pie chart
par(mfrow=c(1,2),mar=c(1,1,1,1))
pie(Uni_Rating_Table,main='University Rating',radius = 1)
pie(Research_Table,main='Research Experience',radius=1)

plot(rawdata)
## Most of the data shows 'positive increase' trend, which means higher the grade, students get higher chance to get acceptance.

# Train/Test
set.seed(55)
new <- rawdata
datatotal <- sort(sample(nrow(new),nrow(new)*0.7))
train <- new[datatotal,]
test <- new[-datatotal,]

# Machine Learning Method
# 1) Logistic Regression
library(caret)
ctrl <- trainControl(method='repeatedcv',repeats=5)
logistic_fit <- train(Chance.of.Admit~.,
                      data=train,
                      method='glm',
                      trControl=ctrl,
                      preProcess=c("center","scale"),
                      metric='RMSE')
l_fit <- 0.05971552

logistic_pred <- predict(logistic_fit,newdata=test)
postResample(pred=logistic_pred,obs=test$Chance.of.Admit)
l_pred <- 0.06069659

# 2) Elastic net Regression
ctrl <- trainControl(method='repeatedcv',repeats=5)
logistic_elastic_fit <- train(Chance.of.Admit~.,
                              data=train,
                              method='glmnet',
                              trControl=ctrl,
                              preProcess=c("center","scale"),
                              metric='RMSE')
l_elas_fit <- 0.05962162

logistic_elastic_pred <- predict(logistic_elastic_fit,newdata=test)
postResample(pred=logistic_elastic_pred,obs=test$Chance.of.Admit)
l_elas_pred <- 0.06068205

# 3) Random Forest
ctrl <- trainControl(method='repeatedcv',repeats=5)
rf_fit <- train(Chance.of.Admit~.,
                data=train,
                method='rf',
                trControl=ctrl,
                preProcess=c("center","scale"),
                metric='RMSE')
r_fit <- 0.06213015

rf_pred <- predict(rf_fit,newdata=test)
postResample(pred=rf_pred,obs=test$Chance.of.Admit)
r_pred <- 0.06105093

# 4) Support Vector Machine
ctrl <- trainControl(method='repeatedcv',repeats=5)
svm_linear_fit <- train(Chance.of.Admit~.,
                        data=train,
                        method='svmLinear',
                        trControl=ctrl,
                        preProcess=c("center","scale"),
                        metric='RMSE')
svm_fit <- 0.06008967

svm_linear_pred <- predict(svm_linear_fit,newdata=test)
postResample(pred=svm_linear_pred,obs=test$Chance.of.Admit)
svm_pred <- 0.06143432

# 5) Kernel Support Vector Machine
ctrl <- trainControl(method='repeatedcv',repeats=5)
svm_poly_fit <- train(Chance.of.Admit~.,
                       data=train,
                       method='svmPoly',
                       trControl=ctrl,
                       preProcess=c("center","scale"),
                       metric='RMSE')
kernel_fit <- 0.05935392

svm_poly_pred <- predict(svm_poly_fit,newdata=test)
postResample(pred=svm_poly_pred,obs=test$Chance.of.Admit)
kernel_pred <- 0.06254307

# Conclusion
final <- data.frame("Type"=c("Train","Test"),"Logistic"=c(l_fit,l_pred),"ElasticNet"=c(l_elas_fit,l_elas_pred),"RandomForest"=c(r_fit,r_pred),"SupportVector"=c(svm_fit,svm_pred),"Kernel"=c(kernel_fit,kernel_pred))

min(final[1,2:6])
## In Train data, 'Kernel' method brings out the smallest RMSE
min(final[2,2:6])
## In Test data, 'Elastic net' method shows the smallest RMSE

## If we compare all of the RMSE values, Elastic net (Test data) has the smallest RMSE. 
## Therefore, if you want to check the probability of getting acceptance from graduate school, use 'Elastic Net Regression Model' in this case since it has the least root mean square error loss.


