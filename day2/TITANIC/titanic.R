##############################
##                          ##
## Titanic ML               ##
##                          ##
## andrea.nigri@uniroma1.it ##
##                          ##
##############################




###################################################################
###################################################################
#library
library(MASS) # for Pima data sets
library(ggplot2)
library(plotROC)
library(readr)
library(caret)
install.packages("devtools")
library(devtools)
devtools::install_github("hadley/ggplot2")
devtools::install_github("sachsmc/plotROC")
library(plotROC)
library(tidyverse)
library(dplyr)
###################################################################
###################################################################

#Data
titanic <- read_csv("titanic_clean.csv")
View(titanic)
head(titanic)
tail(titanic)
titanic <- titanic[-1310,]
tail(titanic)
names(titanic)
str(titanic)

###################################################################
###################################################################
                                                               
## survival - Survival (0 = No; 1 = Yes)                          ##
## class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)            ##
## name - Name                                                    ##
## sex - Sex                                                      ##
## age - Age                                                      ##
## sibsp - Number of Siblings/Spouses Aboard                      ##
## parch - Number of Parents/Children Aboard                      ##
## ticket - Ticket Number                                         ##
## fare - Passenger Fare                                          ##
## cabin - Cabin                                                  ##
## embarked - Port of Embarkation                                 ##
## (C = Cherbourg; Q = Queenstown; S = Southampton)               ##
##                                                                ##
## boat - Lifeboat (if survived)                                  ##
## body - Body number (if did not survive and body was recovered) ##
## home.dest - dest
####################################################################
####################################################################
names(titanic)
titanic <- titanic %>%
  select(survived,pclass,sex,
         age,sibsp,parch,fare,embarked)

table(titanic$survived)
str(titanic)

#titanic <- titanic %>% mutate(pclass=factor(pclass),
#                       sex=factor(sex),survived=factor(survived))
#titanic <- titanic %>% mutate(survived=factor(survived))
table(titanic$survived)

###################################################################
###################################################################
#Data Viz
theme_set(theme_minimal())

titanic %>% ggplot(aes(factor(pclass),fill=factor(sex))) +
  geom_bar(position="dodge")

titanic %>% ggplot(aes(factor(pclass),fill=factor(sex))) +
  geom_bar(position="dodge")+
  facet_grid(".~survived")

titanic %>% ggplot( aes(x = age, fill = factor(survived))) +
  facet_wrap(~pclass) +
  geom_histogram(binwidth = 5) +
  ggtitle("Age by Pclass") + 
  xlab("Age") +
  ylab("Total Count")

titanic %>%  
  ggplot(aes(x = age, fill = factor(survived))) +
  facet_wrap(~sex + pclass) +
  geom_histogram(binwidth = 10) +
  ggtitle("Age by class and gender")+
  xlab("Age") +
  ylab("Total Count")


titanic %>%  
  ggplot(aes(x = age, fill = factor(survived))) +
  facet_wrap(~sex + pclass) +
  geom_density(binwidth = 1) +
  ggtitle("Age by class and gender")+
  xlab("Age") +
  ylab("Total Count")

train.model.ind <- createDataPartition(titanic$survived, p = 0.8, list = FALSE)
train <- titanic[train.model.ind,]
test <- titanic[-train.model.ind,]
train$survived
test$survived

str(train)
str(test)

###################################################################
###################################################################
#LOGIT

names(train)
# train model on training data
glm.out.train <- glm(survived ~pclass+sex+age+sibsp+parch+fare+embarked,
                     data = train,
                     family = binomial)
summary(glm.out.train)

pr.test <- data.frame(pr=predict(glm.out.train, test[,2:8]))

logitpred <- ifelse(pr.test$pr>0.5,1,0)

predroc <- data.frame(logit=logitpred,test=test$survived)
str(predroc)
calculate_roc(logitpred,test$survived,0.05)

basicplot <- ggplot(predroc, aes(d = logit, m =test)) + geom_roc()
basicplot + labs(title="ROC CURVE",subtitle="Logit Vs. Test",
                 x ="False Positive Rate", y = "True Positive Rate")+
  annotate("text", x = .60, y = .55, 
           label = paste("AUC =", round(calc_auc(basicplot)$AUC, 2))) +
  scale_x_continuous("1 - Specificity", breaks = seq(0, 1, by = .1))


###################################################################
###################################################################


#########################
#                       #
# random forest         #
#                       #
#########################


install.packages("randomForest")
install.packages("rattle")
library(randomForest)
library(rpart)
library(rattle)


#fit <- rpart(survived ~ pclass+sex+age+sibsp+parch+
 #              fare+embarked, data = train,
  #           method="class", 
   #          control=rpart.control(minsplit=2, cp=0))

fit <- rpart(survived ~ pclass+sex+age+sibsp+parch+
               fare+embarked, data = train)

plot(fit)

fancyRpartPlot(fit)

pr <- predict(fit,test[,2:8],type = "prob")[,2]

pr.test.tree <- data.frame(pr=pr)

treepred <- ifelse(pr.test.tree$pr>0.5,1,0)

predroc <- data.frame(tree=treepred,test=test$survived)
str(predroc)
calculate_roc(treepred,test$survived,0.05)

basicplot <- ggplot(predroc, aes(d = tree, m =test)) + geom_roc()

basicplot + labs(title="ROC CURVE",subtitle="Tree Vs. Test",
                 x ="False Positive Rate", y = "True Positive Rate")+
  annotate("text", x = .60, y = .55, 
           label = paste("AUC =", round(calc_auc(basicplot)$AUC, 2))) +
  scale_x_continuous("1 - Specificity", breaks = seq(0, 1, by = .1))

