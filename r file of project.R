rm(list=ls())

setwd('E:/data scientist/project edwiser/project 1')
library(psych)
library(ggplot2)
library(dplyr)
library(randomForest)
library(corrplot)
library(corrgram)
library(C50)
library(caret)
library(rlist)
library(RRF)
library(inTrees)
#install.packages("e1071")


testdata=read.csv("test_data.csv")
traindata=read.csv("train_data.csv")

mydata=traindata
mytest=testdata

View(mydata)    
str(mydata)

#----------------------------------------------------Exploratory Data Analysis------------------------------------------------------
dim(mydata)

#change to proper type
colname=colnames(mydata)


mydata$Churn=as.integer(mydata$Churn)
mydata$voice.mail.plan=as.integer(mydata$voice.mail.plan)
mydata$international.plan=as.integer(mydata$international.plan)


mydata$Churn[mydata$Churn=='1']=0
mydata$Churn[mydata$Churn=='2']=1

mydata$international.plan[mydata$international.plan=='1']=0
mydata$international.plan[mydata$international.plan=='2']=1

mydata$voice.mail.plan[mydata$voice.mail.plan=='1']=0
mydata$voice.mail.plan[mydata$voice.mail.plan=='2']=1


mytest$Churn=as.integer(mytest$Churn)
mytest$international.plan=as.integer(mytest$international.plan)
mytest$voice.mail.plan=as.integer(mytest$voice.mail.plan)


mytest$Churn[mytest$Churn=='1']=0
mytest$Churn[mytest$Churn=='2']=1

mytest$international.plan[mytest$international.plan=='1']=0
mytest$international.plan[mytest$international.plan=='2']=1

mytest$voice.mail.plan[mytest$voice.mail.plan=='1']=0
mytest$voice.mail.plan[mytest$voice.mail.plan=='2']=1

mydata$Churn=as.factor(mydata$Churn)
mydata$voice.mail.plan=as.factor(mydata$voice.mail.plan)
mydata$international.plan=as.factor(mydata$international.plan)
mydata$phone.number=as.numeric(mydata$phone.number)

mytest$Churn=as.factor(mytest$Churn)
mytest$voice.mail.plan=as.factor(mytest$voice.mail.plan)
mytest$international.plan=as.factor(mytest$international.plan)
mytest$phone.number=as.numeric(mytest$phone.number)


#----------------------------------------------------Outlier Analysis------------------------------------------------------

numeric_index=sapply(mydata,is.numeric)
numeric_data=mydata[,numeric_index]

num_col_names=colnames(numeric_data[-18])


for(i in 1:length(num_col_names)){
  assign(paste("graph",i),ggplot(aes_string(y= (num_col_names[i]), x= mydata$Churn),data = subset(mydata))+ stat_boxplot(geom = "errorbar",width=.5)+geom_boxplot(outlier.color = "red",fill="grey",outlier.shape = 18,outlier.size = 1,notch = FALSE)+
           labs(y=num_col_names[i],x="churn")+ggtitle(paste("boxplot of",num_col_names[i]) ) )
}

gridExtra::grid.arrange(`graph 1`,`graph 2`,`graph 3`,ncol = 3)
gridExtra::grid.arrange(`graph 4`,`graph 5`,`graph 6`,ncol = 3)
gridExtra::grid.arrange(`graph 7`,`graph 8`,`graph 9`,ncol = 3)
gridExtra::grid.arrange(`graph 10`,`graph 11`,`graph 12`,ncol = 3)
gridExtra::grid.arrange(`graph 13`,`graph 14`,`graph 15`,ncol = 3)
gridExtra::grid.arrange(`graph 16`,`graph 17`,ncol = 2)

for (i in num_col_names) {
  print(i)  
  maxi=quantile(mydata[,i],c(.75))+(1.5*IQR(mydata[,i]))
  mini=quantile(mydata[,i],c(.75))-(1.5*IQR(mydata[,i]))
  
  #val=mydata[,i][fulldata[,i] %in% boxplot.stats(mydata[,i])$out]    
  
  #mydata[,i][mydata[,i] %in% val]=last
  mydata[,i][mydata[,i] > maxi]=maxi
  mydata[,i][mydata[,i] <mini]=mini
}


#----------------------------------------------------Features Selection------------------------------------------------------
#correlation matrix for numeric varables

corrgram(mydata[sapply(mydata, is.numeric)])
cor(mydata$total.day.minutes,mydata$total.day.charge)
cor(mydata$total.eve.minutes,mydata$total.eve.charge)
cor(mydata$total.night.minutes,mydata$total.night.charge)
cor(mydata$total.intl.minutes,mydata$total.intl.charge)


#chi-square test for categorial varable
cvar=sapply(mydata, is.factor)
cdata=mydata[,cvar]
cdata
for (u in 1:3) {
  print(names(cdata[u]))
  print(chisq.test(table(cdata$Churn,cdata[,u]))) 
}

# drop unneeded varables
mydata=subset(mydata,select= -c(total.night.charge,total.day.charge,total.intl.charge,total.eve.charge))
mytest=subset(mytest,select= -c(total.night.charge,total.day.charge,total.intl.charge,total.eve.charge))
str(mydata)

#**************************************************************************************************
#                                  Decision Tree model using C50
#***************************************************************************************************
set.seed(1234)
model_C5.0=C5.0(Churn ~ ., data = mydata, rules = TRUE)
summary(model_C5.0)
pred_value=predict(model_C5.0, mytest[-17])

conf_matrix=table(mytest$Churn,pred_value)
conf_matrix

confusionMatrix(conf_matrix)

# Fnr = .3303  accuracy = .9454


#*************************************************************************************************
#                                   Randam Forest Algo
#**************************************************************************************************

model_randamforest= randomForest(Churn ~ . ,mydata, importance= TRUE, ntree= 2000)

treelist=RF2List(model_randamforest)
exec=extractRules(treelist,mydata[,-17])
exec[1:2,]
readable=presentRules(exec,colnames(mydata))
readable[1:2,]
targ=getRuleMetric(exec,mydata[,-18],mydata$Churn)
targ[1:2,]

rr=presentRules(targ,colnames(mydata))
rr[1:2,]

pred_rf_value=predict(model_randamforest, mytest[,-17])

conf_matrix_rf=table(mytest$Churn,pred_rf_value)
conf_matrix_rf

confusionMatrix(conf_matrix_rf)


# Fnr = .2589  accuracy = .9634


#*********************************************************************************************************
#                                      Logistic Regression
#*******************************************************************************************************

model_logistic=glm(Churn ~. , data = mydata, family = "binomial")
summary(model_logistic)
pred_logit=predict(model_logistic, newdata = mytest,type = "response")

pred_logit

pred_logit=ifelse(pred_logit > 0.5, 1, 0)

conf_matrix_logit=table(mytest$Churn,pred_logit)
conf_matrix_logit

confusionMatrix(conf_matrix_logit)

#fnr=.7973   accuracy = .874


