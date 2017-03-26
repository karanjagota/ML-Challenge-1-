library(data.table)
library(xgboost)

xx = fread("train.csv")
test = fread("test.csv")

target<- xx$loan_status
train_member_id<- xx$member_id
test_member_id<- test$member_id
xx$loan_status<-NULL

train<-rbind(xx,test) 
  
train[,c('funded_amnt','funded_amnt_inv','collection_recovery_fee'):=NULL]

train$term<- ifelse(train$term =="36 months",0,1)
train$emp_length<- as.numeric(as.factor(train$emp_length))
train$new_var<- ifelse(train$desc=="",0,1)
train$initial_list_status<- as.numeric(as.factor(train$initial_list_status))-1
check_pattern <- colnames(train)[sapply(train, is.numeric)]
check_pattern <- check_pattern[!(check_pattern %in% c("member_id"))]


skew <- sapply(train[,check_pattern,with=F], function(x) skewness(x,na.rm = T))
skew <- skew[skew > 2]
skew<- skew[-1]

train[,(names(!is.na(skew))) := lapply(.SD, function(x) log(x + 10)), .SDcols = names(!is.na(skew))]
train[,dti := log10(dti + 10)]
train[,pymnt_plan := NULL]
train[,verification_status_joint :=  NULL]
train[,application_type := NULL]
train[,title := NULL]
train[,batch_enrolled := NULL]

train$last_week_pay<- as.numeric(as.factor(train$last_week_pay))
train$sub_grade<-as.numeric(as.factor(train$sub_grade))
train$verification_status<- as.numeric(as.factor(train$verification_status))
train$home_ownership<- as.numeric(as.factor(train$home_ownership))
train$purpose<- as.numeric(as.factor(train$purpose))
train$grade<- as.numeric(as.factor(train$grade))
train$addr_state<- as.numeric(as.factor(train$addr_state))
train$zip_code<- as.numeric(as.factor(train$zip_code))
train$emp_title<-as.numeric(as.factor(train$emp_title))
train$desc<-NULL

xx_train <- train[1:532428,]
xx_test <- train[532429:887379,]

xx_train[is.na(xx_train)]<- -1
xx_test[is.na(xx_test)]<- -1

xx_train<- xx_train[,-1]
xx_test<- xx_test[,-1]

xx_train$collections_12_mths_ex_med<- NULL
xx_train$addr_state<- NULL
xx_train$purpose<- NULL
xx_test$collections_12_mths_ex_med<-NULL
xx_test$addr_state<-NULL
xx_test$purpose<-NULL

set.seed(488)
bst <- xgboost(data=as.matrix(xx_train), label=as.matrix(target), objective="binary:logistic", nrounds=200, eta=0.05, max_depth=6, subsample=0.75, colsample_bytree=0.8, min_child_weight=1, eval_metric="auc")

pred <- predict(bst, as.matrix(xx_test))

a<- data.frame(member_id = test_member_id,loan_status=pred)
write.csv(a,"model_91.csv",row.names = F)
