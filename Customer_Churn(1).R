rm(list = ls())
#set working darectory
setwd("D:/projects/Project")
#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')
#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

#Read the data
df_train= read.csv("D:/projects/Project/Train_data.csv", header = TRUE)
head(df_train)

# get the summary of the data
summary(df_train)

##Data Manupulation; convert string categories into factor numeric

for (i  in 2:ncol(df_train)){
  if(class(df_train[, i])== "factor"){
    df_train[, i]= factor(df_train[,i], labels = (1: length(levels(factor(df_train[,i])))))
  }
  
}

df_train$num._customer_service_calls= as.factor(df_train$num._customer_service_calls)
df_train$number_vmail_msg= as.factor(df_train$number_vmail_msg)


##########################OUTLIER ANALYSIS###################################################
numeric_index = sapply(df_train,is.numeric) #selecting only numeric
numeric_data = df_train[,numeric_index]

#Calculate Absolute Z-score
for (i in 1:length(numeric_data)) {
  #print(i)
  stdev <- sqrt(sum((df_train[, i] - mean(df_train[, i], na.rm = T))^2,
                    na.rm = T) / sum(!is.na(df_train[, i])))
  absZ <- abs(df_train[, i] - mean(df_train[, i], na.rm = T)) / stdev
}
#Consider z > 3 as an outlier and remove it
df_train= df_train[absZ < 3,]

#####################EXPLORATERY DATAANALYSIS##############################################

# prep frequency table
freqtable <- table(df_train$state)
df_state <- as.data.frame.table(freqtable)
names(df_state)[names(df_state) == "Var1"] <- "States"
df_state <- df_state[order(df_state$Freq, decreasing = TRUE), ]
df_state$States <- factor(df_state$States,
                          levels = df_state$States)  # to retain the order in plot.
head(df_state)

library(ggplot2)
theme_set(theme_classic())

g <- ggplot(df_state, aes(States, Freq))
g + geom_bar(stat="identity", width = 0.5, fill="tomato2") + 
  labs(title="Bar Chart", 
       subtitle="Users in Each State") +
  theme(axis.text.x = element_text(angle=65, vjust=0.6))
colnames(df_train)

# Show relationship of churn and state statistically

xtabs(~ state + Churn, data = df_train)
g <- ggplot(df_train, aes(state))
g + geom_bar(aes(fill=Churn), width = 0.5) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) +
  labs(title="Churn in Each State")

g <- ggplot(df_train, aes(voice_mail_plan))
g + geom_bar(aes(fill=Churn), width = 0.5) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) +
  labs(title="Churn against voice mail plan")
 
#higher Churn in higher daycharge
g <- ggplot(df_train, aes(Churn, total_day_charge))
g + geom_boxplot(varwidth=T) +
  coord_flip()+
  labs(title="Box plot", 
       subtitle="Day Charge and CHurn",
       x="Curn",
       y="Day Charge")

# For churning class yes eve charge is also higher but less compared to day charge
g <- ggplot(df_train, aes(Churn, total_eve_charge))
g + geom_boxplot(varwidth=T) + 
  labs(title="Box plot", 
       subtitle="Eve Charge and CHurn",
       x="Curn",
       y="Day Charge")

#not much influence by night charge on costumer churning
g <- ggplot(df_train, aes(Churn, total_nightcharge))
g + geom_boxplot(varwidth=T) + 
  labs(title="Box plot", 
       subtitle="Night Charge and CHurn",
       x="Curn",
       y="Night Charge")


#num._customer_service_calls and Churn
xtabs(~ num._customer_service_calls + Churn, data = df_train)
ggplot(df_train, aes(x =num._customer_service_calls , fill = Churn)) + 
  geom_bar() 
  
   
  
##Feature Selection
library(ggcorrplot)

corr= round(cor(numeric_data), 1)
ggcorrplot(corr, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("tomato2", "white", "springgreen3"), 
           title="Correlogram of Churn", 
           ggtheme=theme_bw)
#Chi Sqare
factor_index = sapply(df_train,is.factor)
factor_data = df_train[,factor_index]

for (i in 1:length(factor_data))
{
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$Churn,factor_data[,i])))
}

## Dimension Reduction
df_train = subset(df_train, 
                         select = -c(phone_num, area_code, total_day_minutes, total_eve_mnts,
                                     total_night_minutes, total_intl_minutes, account_len))


#Standardisation
num_index= sapply(df_train, is.numeric)
cnames= colnames(df_train[, num_index])
c_sub= subset(cnames, select= c(total_day_calls, total_day_charge, total_eve_calls, 
                                 total_eve_charge, total_night_calls, total_nightcharge,
                                 total_intl_calls, total_intl_charge))
head(cnames)
for(i in cnames){
  print(i)
  df_train[,i] = (df_train[,i] - mean(df_train[,i]))/sd(df_train[,i])
}

head(df_train)

###################MODELING###########################
install.packages("fastDummies")
library(fastDummies)
df_train <-dummy_cols(df_train,  select_columns = "state")
df_train <- subset(df_train, select = -state)

#Divide data into train and test using stratified sampling method
set.seed(1234)
##Stratified Sampling
library(caret)
train.rows<- createDataPartition(y= df_train$Churn, p=0.8, list = FALSE)
train_data<- df_train[train.rows,] # 80% data goes in here
test_data<- df_train[-train.rows,]
table(train_data$Churn)

##Decision tree for classification
#Develop Model on training data
library(C50)
C50_model = C5.0(Churn ~., train_data, trials = 100, rules = TRUE)

#Summary of DT model
summary(C50_model)

#Lets predict for test cases
library(ROSE) #for accu.meas and roc.curve function
C50_Predictions = predict(C50_model, test_data[,-13], type = "class")
accuracy.meas(test_data$Churn, C50_Predictions)
roc.curve(test_data$Churn, C50_Predictions, plotit = F)
summary(C50_Predictions)


##Evaluate the performance of classification model
library(e1071)
ConfMatrix_C50 = table(test_data$Churn, C50_Predictions)
confusionMatrix(ConfMatrix_C50)
roc.curve(test_data$Churn, C50_Predictions, plotit = F)


#False Negative rate
#FNR = FN/FN+TP= 
FNR= 63/(63+505)             #  0.14

library(randomForest)
RF_model = randomForest(Churn ~ ., train_data, importance = TRUE, ntree = 500)
#Extract rules fromn random forest
#transform rf object to an inTrees' format
library(inTrees)
treeList = RF2List(RF_model)

exec = extractRules(treeList, train_data[,-13])
exec[1:2,]

readableRules = presentRules(exec, colnames(train_data))

#Get rule metrics
ruleMetric = getRuleMetric(exec, train_data[,-13], train_data$Churn)  # get rule metrics
#evaulate few rules
ruleMetric[1:10,]
#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test_data[,-13])

##Evaluate the performance of classification model
ConfMatrix_RF = table(test_data$Churn, RF_Predictions)
confusionMatrix(ConfMatrix_RF)
FNR = FN/FN+TP
6/(6+562)       #0.01
roc.curve(test_data$Churn, RF_Predictions, plotit = F)

###########UNDERSAMPLE##############
table(train_data$Churn)
data_balanced_under <- ovun.sample(Churn ~ ., data = train_data, method = "under",
                                   N = 900, seed = 1)$data
table(data_balanced_under$Churn)
library(C50)
C50_model = C5.0(Churn ~., data_balanced_under, trials = 100, rules = TRUE)
#Summary of DT model
summary(C50_model)

#Lets predict for test cases
C50_Predictions_und = predict(C50_model, test_data[,-13], type = "class")
accuracy.meas(test_data$Churn, C50_Predictions)
roc.curve(test_data$Churn, C50_Predictions, plotit = T)   #(AUC): 0.83

ConfMatrix_C50_und = table(test_data$Churn, C50_Predictions_und)
confusionMatrix(ConfMatrix_C50_und)
roc.curve(test_data$Churn, C50_Predictions_und, plotit = T)

###########RANDOM FOREST AFTER UNDER SAMPLING####################

RF_model_under = randomForest(Churn ~ ., data_balanced_under, importance = TRUE, ntree = 500)

#Predict test data using random forest model
RF_Predictions_und = predict(RF_model_under, test_data[,-14])

#Create confusion matrix 
ConfMatrix_RF_und = table(test_data$Churn, RF_Predictions_und)
confusionMatrix(ConfMatrix_RF_und)

#ROC curve
roc.curve(test_data$Churn,RF_Predictions_und, plotit = F)


########BOTH UNDER SAMPLE AND OVERSAMPPLE############

data_balanced_both <- ovun.sample(Churn ~ ., data = train_data, 
                                  method = "both", p=0.5,
                                  N=3322, seed = 1)$data
table(train_data$Churn)
table(data_balanced_both$Churn)

#####TREE FOR BOTH BALANCE DATA##################
C50_model_both = C5.0(Churn ~., data_balanced_both, trials = 100, rules = TRUE)

#Summary of DT model
summary(C50_model)

#Lets predict for test cases
C50_Predictions_both = predict(C50_model_both, test_data[,-14], type = "class")
accuracy.meas(test_data$Churn, C50_Predictions)

roc.curve(test_data$Churn, C50_Predictions, plotit = T) #(AUC): 0.83

ConfMatrix_C50_both = table(test_data$Churn, C50_Predictions_both)
confusionMatrix(ConfMatrix_C50_both)
roc.curve(test_data$Churn, C50_Predictions_und, plotit = F)#AUC= 0.885

# Random forest for both under and over sampling

RF_model_both = randomForest(Churn ~ ., data_balanced_both, importance = TRUE, ntree = 500)

#Predict test data using random forest model
RF_Predictions_both = predict(RF_model_both, test_data[,-14])

#Create confusion matrix 
ConfMatrix_RF_both = table(test_data$Churn, RF_Predictions_both)
confusionMatrix(ConfMatrix_RF_both)
#ROC curve
roc.curve(test_data$Churn,RF_Predictions_both, plotit = T) #(AUC): 0.84


#over sampling
data_balanced_over <- ovun.sample(Churn ~ ., data = train_data,
                                   method = "over",
                                   N = 3322)$data

RF_model_over = randomForest(Churn ~ ., data_balanced_over,
                              importance = TRUE, ntree = 500)

#Predict test data using random forest model
RF_Predictions_over = predict(RF_model_over, test_data[,-13])

#Create confusion matrix 
ConfMatrix_RF_over= table(test_data$Churn, RF_Predictions_over)
confusionMatrix(ConfMatrix_RF_over)

#ROC curve
roc.curve(test_data$Churn,RF_Predictions_over, plotit = T)













