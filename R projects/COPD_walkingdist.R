# This is a fictitious dataset obtained from the 2nd course of the Statistical Analysis with R for Public Health specialization: Linear Regression in R for public health.
# The outcome variable of this analysis is MWT1Best, a measure of walking distance. 
# we want to identify factors that predict walking ability in patients with COPD and produce a regression model that will best quantify the impact of these factors.

#loading the relevant libraries
library(tidyverse)
library(Hmisc)
library(gmodels)
library(mctest)

#loading dataset
data <- read_csv("C:\\Users\\isaac\\OneDrive\\Documents\\Projects\\datasets\\COPD.csv") %>%
  rename_all(tolower)

#examining data
head(data)
str(data)
describe(data)

# the sample size is n = 101
# Missing values found in mwt1, mwt2, mwt1best
# continuous vairables: age, packhistory, mwt1, mwt2, mwt1best, fev1, fev1pred, fvc, fvcpred, cat, had, sgrq
# categorical variables: copdseverity, agequartiles, copd, smoking
# binary variables:gender, diabetes, muscular, hypertension, atrialfib, ihd

#changing categorical variables to factors
data <- data %>% mutate(
  copdseverity = as.factor(copdseverity),
  copd = as.factor(copd),
  agequartiles = as.factor(agequartiles),
  gender = as.factor(gender),
  smoking = as.factor(smoking),
  diabetes = as.factor(diabetes),
  muscular = as.factor(muscular),
  hypertension = as.factor(hypertension),
  atrialfib = as.factor(atrialfib),
  ihd = as.factor(ihd)
) %>%
  select(-c(...1, id))
  

#examining continuous variables individually
summary(data$age)
hist(data$age, main = "Age")

summary(data$packhistory)
hist(data$packhistory, main = "Packhistory")

summary(data$mwt1)
hist(data$mwt1, main = "mwt1")

summary(data$mwt2)
hist(data$mwt2, main = "mwt2")

summary(data$mwt1best)
hist(data$mwt1best, main = "mwt1best")

summary(data$fev1)
hist(data$fev1, main = "fev1")

summary(data$fev1pred)
hist(data$fev1pred, main = "fev1pred")

summary(data$fvc)
hist(data$fvc, main = "fvc")

summary(data$fvcpred)
hist(data$fvcpred, main = "fvcpred")

summary(data$sgrq)
hist(data$sgrq, main = "sgrq")

summary(data$cat)
hist(data$cat, main = "cat")
#note that there is an outlier in CAT
#copd assessment test (CAT) scores should range between 0 to 40, but in this sample there is a max value of 188.

summary(data$had)
hist(data$cat, main = "had")
# note that there is an outlier in HAD
# hospital anxiety and depression scale(HADS) shuold range between 0 to 21, but in this sample there is a max value of 56.2

# In a research study we would always approach the researcher to clarify whether the outlier is an error in data entry. 
# sometimes the error can be rectified by replacing the erroneous data with the accurate one. 
# However, since this dataset is fictitious, the incorrect data will be removed.

#removing false values
data$had[data$had > 21] <- NA
data$cat[data$cat > 40] <- NA


#re-examining data after erroneous data removed
summary(data$cat)
hist(data$cat, main = "cat")

summary(data$had)
hist(data$had, main = "had")




#examining categorical and binary variables individually
describe(data$copdseverity)
describe(data$copd)
describe(data$agequartiles)
describe(data$gender)
describe(data$smoking) #note that there are only 2 distinct values: ex- smokers and current smokers. there are no non-smokers in this sample
describe(data$diabetes)
describe(data$muscular)
describe(data$hypertension)
describe(data$atrialfib)
describe(data$ihd)

#noted that copdseverity and copd variables are duplicates. Hence, removing  copdseverity.
## mwt1 and mwt2 will also be removed as they are measures that directly determine mwt1best (highest value from these 2 variables = mwt1best).
data <- data %>% select(-c(copd, mwt1, mwt2))


# Due to limitations of sample size, comorbid conditions cannot be analysed as individual predictors. 
# They will be combined to a single binomial variable "comorbid" to reduce the number of predictors and avoid overfitting.
data <- data %>% mutate(comorbid = ifelse(
  diabetes == 0 & 
  muscular == 0 &
  hypertension == 0 &
  atrialfib == 0 &
  ihd == 0,
  0, 1)) %>%
    select(-c(diabetes, muscular, hypertension, atrialfib, ihd))

data$comorbid <- as.factor(data$comorbid)

#examining comorbid variable
describe(data$comorbid)


#fitting simple linear regression models between the outcome variable(MTW1Best) and each candidate predictor variable
plot(data$age, data$mwt1best)
lr1 <- lm(mwt1best ~ age, data, main = "Age")
summary(lr1)
confint(lr1)
abline(lr1, col = "blue", lwd = 2)

plot(data$packhistory, data$mwt1best, main = "Pack history")
lr2 <- lm(mwt1best ~ packhistory, data)
summary(lr2)
confint(lr2)
abline(lr2, col = "blue", lwd = 2)

plot(data$copdseverity, data$mwt1best, main = "COPD Severity")
lr3 <- lm(mwt1best ~ copdseverity, data)
summary(lr3)
confint(lr3)

plot(data$fev1, data$mwt1best, main = "FEV1")
lr4 <- lm(mwt1best ~ fev1, data)
summary(lr4)
confint(lr4)
abline(lr4, col = "blue", lwd = 2)

plot(data$fev1pred, data$mwt1best, main = "Predicted FEV1")
lr5 <- lm(mwt1best ~ fev1pred, data)
summary(lr5)
confint(lr5)
abline(lr5, col = "blue", lwd = 2)

plot(data$fvc, data$mwt1best, main = "FVC")
lr6 <- lm(mwt1best ~ fvc, data)
summary(lr6)
confint(lr6)
abline(lr6, col = "blue", lwd = 2)

plot(data$fvcpred, data$mwt1best, main = "Predicted FVC")
lr7 <- lm(mwt1best ~ fvcpred, data)
summary(lr7)
confint(lr7)
abline(lr7, col = "blue", lwd = 2)

plot(data$cat, data$mwt1best)
lr8 <- lm(mwt1best ~ cat, data)
summary(lr8)
confint(lr8)
abline(lr8, col = "blue", lwd = 2)

plot(data$had, data$mwt1best, main = "HAD")
lr9 <- lm(mwt1best ~ had, data)
summary(lr9)
confint(lr9)
abline(lr9, col = "blue", lwd = 2)

plot(data$sgrq, data$mwt1best, main = "SGRQ")
lr10 <- lm(mwt1best ~ sgrq, data)
summary(lr10)
confint(lr10)
abline(lr10, col = "blue", lwd = 2)

plot(data$agequartiles, data$mwt1best)
lr11 <- lm(mwt1best ~ agequartiles, data)
summary(lr11)
confint(lr11)

plot(data$gender, data$mwt1best)
lr12 <- lm(mwt1best ~ gender, data)
summary(lr12)
confint(lr12)

plot(data$smoking, data$mwt1best)
lr13 <- lm(mwt1best ~ smoking, data)
summary(lr13)
confint(lr13)

plot(data$comorbid, data$mwt1best)
lr14 <- lm(mwt1best ~ comorbid, data)
summary(lr14)
confint(lr14)


# variables that were statistically significantly associated with walking distance:
# age, packhistory, copdseverity, fev1, fev1pred, fvc, fvcpred , cat, had, sgrq, comorbid

# since fev1, fev1pred, fvc and fvcpred are all measures of lung volume, it is expected that they will be highly correlated.
# CAT and SGRQ are also expected to be correlated as they are both measures of COPD severity.
# this is confirmed by examining the correlation matrix

continuous <- data[, c("age", "packhistory","fev1", "fev1pred", "fvc", "fvcpred", "cat", "had", "sgrq")]
cor_matrix <- cor(continuous, method = "spearman", use = "complete.obs")
cor_matrix
pairs(~age + packhistory + fev1 + fev1pred + fvc + fvcpred + cat + had + sgrq, data = data)

# to avoid multicollinearity, variables that are highly correlated must be excluded from the multiple linear regression model.
# Amongst the different measures of lung function, fev1 explains the most variance. Therefore it will be included in the final model.

# as CAT, SGRQ and copdseverity are all measures of COPD severity, only one of these will be included in the multiple linear regression model.
# Amongst these measures, SGRQ explains the most variance. Therefore it will be included in the final model.

# packhistory and smoking, age and agequartiles are also likely to be collinear.
# packhistory and age will be favoured for inclusion in the multiple linear regression model as they provide more information.


# fitting the multiple linear regression model
mlr1<- lm(mwt1best ~ age + packhistory + fev1 + had + sgrq + comorbid, data)
summary(mlr1)
confint(mlr1)
plot(mlr1)

#checking for collinearity
imcdiag(mod = mlr1, method = "VIF")
