library(ggplot2)
library(caret)
library(caretEnsemble)
library(ROSE)
library(mlbench)
library(DMwR)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

#Load the dataset
getwd()
setwd("D:\\Spring 2019\\SCM 649\\Project")
mydata <- read.csv("bank-full.csv" , sep = ";")

#Summary on dataset
summary(mydata)

str(mydata)
head(mydata)

p_age <- ggplot(mydata, aes(factor(y), age)) + geom_boxplot(aes(fill = factor(y)))
p_age

p_balance <- ggplot(mydata, aes(factor(y), balance)) + geom_boxplot(aes(fill = factor(y)))
p_balance

p_day <- ggplot(mydata, aes(factor(y), day)) + geom_boxplot(aes(fill = factor(y)))
p_day

p_duration <- ggplot(mydata, aes(factor(y), duration)) + geom_boxplot(aes(fill = factor(y)))
p_duration

p_campaign <- ggplot(mydata, aes(factor(y), campaign)) + geom_boxplot(aes(fill = factor(y)))
p_campaign

p_pdays <- ggplot(mydata, aes(factor(y), pdays)) + geom_boxplot(aes(fill = factor(y)))
p_pdays

p_previous <- ggplot(mydata, aes(factor(y), previous)) + geom_boxplot(aes(fill = factor(y)))
p_previous

#Generate dummy variables
for(level in unique(mydata$job)){
  mydata[paste("job", level, sep = "_")] <- ifelse(mydata$job == level, 1, 0)
}

for(level in unique(mydata$marital)){
  mydata[paste("marital", level, sep = "_")] <- ifelse(mydata$marital == level, 1, 0)
}

for(level in unique(mydata$education)){
  mydata[paste("education", level, sep = "_")] <- ifelse(mydata$education == level, 1, 0)
}

mydata$default_yes <- ifelse(mydata$default == "yes", 1, 0)

mydata$housing_yes <- ifelse(mydata$housing == "yes", 1, 0)

mydata$loan_yes <- ifelse(mydata$loan == "yes", 1, 0)

for(level in unique(mydata$contact)){
  mydata[paste("contact", level, sep = "_")] <- ifelse(mydata$contact == level, 1, 0)
}

for(level in unique(mydata$month)){
  mydata[paste("month", level, sep = "_")] <- ifelse(mydata$month == level, 1, 0)
}

for(level in unique(mydata$poutcome)){
  mydata[paste("poutcome", level, sep = "_")] <- ifelse(mydata$poutcome == level, 1, 0)
}

mydata$Class <- ifelse(mydata$y == "yes", "Yes", "No")

#Remove unwanted columns
mydata$X <- NULL
mydata$job <- NULL
mydata$marital <- NULL
mydata$education <- NULL
mydata$default <- NULL
mydata$housing <- NULL
mydata$loan <- NULL
mydata$contact <- NULL
mydata$month <- NULL
mydata$poutcome <- NULL
mydata$y <- NULL

mydata$Class <- as.factor((mydata$Class))
colnames(mydata)[11] <- "job_blue_collar"
colnames(mydata)[14] <- "job_admin"
colnames(mydata)[16] <- "job_self_employeed"

#Splitting
set.seed(1)
training_size <- floor(0.80 * nrow(mydata))
train_ind <- sample(seq_len(nrow(mydata)), size = training_size)
training <- mydata[train_ind, ]
testing <- mydata[-train_ind, ]

#Normalizing
preProcValues <- preProcess(training, method = c("center", "scale"))
scaled.training <- predict(preProcValues, training)
scaled.testing <- predict(preProcValues, testing)

#Sampling
ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

set.seed(2)
down_training <- downSample(x = scaled.training[, -ncol(scaled.training)],
                            y = scaled.training$Class)

up_training <- upSample(x = scaled.training[, -ncol(scaled.training)],
                        y = scaled.training$Class)

smote_training <- SMOTE(Class~., data = scaled.training)

rose_training <- ROSE(Class~., data = scaled.training, seed=2)$data

#Model training - CART
set.seed(3)
orig_fit <- train(Class~., data = training, 
                  method = "rpart",
                  metric = "ROC",
                  trControl = ctrl)

set.seed(4)
down_outside <- train(Class~., data = down_training, 
                      method = "rpart",
                      metric = "ROC",
                      trControl = ctrl)

set.seed(5)
up_outside <- train(Class~., data = up_training, 
                    method = "rpart",
                    metric = "ROC",
                    trControl = ctrl)

set.seed(6)
smote_outside <- train(Class~., data = smote_training, 
                       method = "rpart",
                       metric = "ROC",
                       trControl = ctrl)

set.seed(7)
rose_outside <- train(Class~., data = rose_training, 
                      method = "rpart",
                      metric = "ROC",
                      trControl = ctrl)

#Model testing - Original
original_model <- list(original = orig_fit)

test_roc <- function(model, data) {
  library(pROC)
  roc_obj <- roc(data$Class, 
                 predict(model, data, type = "prob")[, "Yes"],
                 levels = c("No", "Yes"))
  ci(roc_obj)
}

original_test <- lapply(original_model, test_roc, data = testing)
original_test <- lapply(original_test, as.vector)
original_test <- do.call("rbind", original_test)
colnames(original_test) <- c("lower", "ROC", "upper")
original_test <- as.data.frame(original_test)

#Model testing - Resampled
scaled_models <- list(down = down_outside,
                      up = up_outside,
                      SMOTE = smote_outside,
                      ROSE = rose_outside)

scaled_test <- lapply(scaled_models, test_roc, data = scaled.testing)
scaled_test <- lapply(scaled_test, as.vector)
scaled_test <- do.call("rbind", scaled_test)
colnames(scaled_test) <- c("lower", "ROC", "upper")
scaled_test <- as.data.frame(scaled_test)

cart_test <- rbind(original_test,scaled_test)

fancyRpartPlot(up_outside$finalModel)

