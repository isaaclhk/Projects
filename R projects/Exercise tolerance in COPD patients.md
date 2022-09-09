# Predictors of Exercise Tolerance in Patients with COPD
## Background
This is a fictitious dataset obtained from the 2nd course of the Statistical Analysis with R for Public Health specialization: Linear Regression in R for public health.
The dataset can be found [here](https://github.com/isaaclhk/Projects/blob/main/datasets/COPD.csv).
The outcome variable for this analysis is MWT1Best, a measure of walking distance. 
we want to identify factors that predict walking ability in patients with COPD and produce a regression model that will best quantify the impact of these factors.

### Variables in the dataset
1. Age - Age in years
2. Packhistory - number of packs of cigarettes smoked per day * number of smoking years
3. COPD severity- levels = c(mild, moderate, severe, very severe)
4. MWT1 - Distance covered over first attempt of 6-minute walking test
5. MWT2- Distance covered over second attempt of 6- minute walking test
6. MWT1Best - Best performance between MWT1 and MWT2
7. FEV1- forced expiratory volume for the first second of exhalation
8. FEV1PRED - percentage of FEV1 relative to predicted value
9. FVC- forced vital capacity (total volume of air exhaled during FEV)
10. FVCPRED- percentage of FVC relative to predicted value
11. CAT- COPD Assessment Test, a questionnaire to quantify the impact of COPD on a person's life
12. HAD- Hospital Anxiety and Depression Scale, a questionnaire to measure anxiety and depression
13. SGRQ- St George's Respiratory Questionnaire, measures impact on daily life and perceived well-being in patients with obstructive airways disease.
14. AGEquartiles- Age variable categorised into quartiles
15. COPD- Indication of COPD severity, levels = 1:4 (mild to very severe)
16. Gender- levels = 0(female), 1(male)
17. Smoking- levels = 0(non-smokers), 1(ex-smokers), 2(current smokers)
18. Other comorbidities e.g. Diabetes, muscular, hypertension, AtrialFib, IHD- levels - 0(absent), 1(present)

Continuous variables: age, packhistory, mwt1, mwt2, mwt1best, fev1, fev1pred, fvc, fvcpred, cat, had, sgrq
Categorical variables: copdseverity, agequartiles, copd, smoking
Binary variables: gender, diabetes, muscular, hypertension, atrialfib, ihd

## Code
First, the relevant libraries and dataset are loaded. The column names are changed to all lowercase letters for simplicity, 
then we have a brief overview of the dataset.

```
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
```

</br>
After examining the dataset's structure, we note a few important points:
1. The sample size is n = 101
2. Missing values are present in mwt1, mwt2, and mwt1best
3. most variables are numeric, except for copdseverity which is character.

Missing values are a common problem. In a research study, we can always approach the researcher to ask them about to the missing value. 
However, since this dataset is fictitious, the missing value will be left empty. </br> </br>

The categorical and binary variables will be changed to factors, and the first two columns of the dataset will be removed as they are not required.
</br>

```
#changing categorical variables to factors
data <- data %>% mutate(
  copdseverity = as.factor(copdseverity),
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

```

</br>
Each continuous variable is examined individually. This will give us a sense of the variable's distribution and help us to identify outliers.
</br>

```
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

summary(data$had)
hist(data$cat, main = "had")

#removing false values
data$had[data$had > 21] <- NA
data$cat[data$cat > 40] <- NA
```

</br>
Note that there is an outlier in CAT.
copd assessment test (CAT) scores should range between 0 to 40, but in this sample there is a max value of 188.
</br>
Note that there is also an outlier in HAD.
hospital anxiety and depression scale(HADS) shuold range between 0 to 21, but in this sample there is a max value of 56.2
</br> </br>
In a research study, we would always approach the researcher to clarify whether the outlier is an error in data entry. 
sometimes the error can be rectified by replacing erroneous data with the accurate one. 
However, since this dataset is fictitious, the incorrect data will be removed.
</br>

```
#removing false values
data$had[data$had > 21] <- NA
data$cat[data$cat > 40] <- NA

#re-examining data after erroneous data removed
summary(data$cat)
hist(data$cat, main = "cat")

summary(data$had)
hist(data$had, main = "had")
```

</br>
As with the continuous variables, the categorical and binary variables are also examined.
</br>

```
describe(data$copdseverity)
describe(data$copd)
describe(data$agequartiles)
describe(data$gender)
describe(data$smoking) 
describe(data$diabetes)
describe(data$muscular)
describe(data$hypertension)
describe(data$atrialfib)
describe(data$ihd)
```

</br> 
After looking at the data, we note that:
1. there are only 2 distinct values: ex- smokers and current smokers. there are no non-smokers in this sample.
2. copd and copdseverity are duplicates.
</br> </br>
Hence, copdseverity will be removed. mwt1 and mwt2 will also be removed as they are measures that directly determine the outcome mwt1best, so they should not be considered as candidate predictors.
</br>

```
data <- data %>% select(-c(copd, mwt1, mwt2))
```

</br>
Due to limitations of sample size, comorbid conditions cannot be analysed as individual predictors. 
They will be combined to a single binomial variable "comorbid" to reduce the number of predictors and avoid overfitting.
</br>

```
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
```

</br>
The relationships between each candidate predictor variable and the outcome variable is analysed individually by fitting simple linear regression models.
This informs us about what we should expect in the final regression model. Moreover, if large differences in the coefficients arise between the results of the simple linear regressions and the multiple linear regression model, it may hint at multicollinearity between predictors in the multiple linear regression model.
</br>

```
#fitting simple linear regression models between the outcome variable(MTW1Best) and each candidate predictor variable
lr1 <- lm(mwt1best ~ age, data)
summary(lr1)
confint(lr1)


lr2 <- lm(mwt1best ~ packhistory, data)
summary(lr2)
confint(lr2)


lr3 <- lm(mwt1best ~ copdseverity, data)
summary(lr3)
confint(lr3)


lr4 <- lm(mwt1best ~ fev1, data)
summary(lr4)
confint(lr4)

lr5 <- lm(mwt1best ~ fev1pred, data)
summary(lr5)
confint(lr5)

lr6 <- lm(mwt1best ~ fvc, data)
summary(lr6)
confint(lr6)

lr7 <- lm(mwt1best ~ fvcpred, data)
summary(lr7)
confint(lr7)

lr8 <- lm(mwt1best ~ cat, data)
summary(lr8)
confint(lr8)

lr9 <- lm(mwt1best ~ had, data)
summary(lr9)
confint(lr9)

lr10 <- lm(mwt1best ~ sgrq, data)
summary(lr10)
confint(lr10)

lr11 <- lm(mwt1best ~ agequartiles, data)
summary(lr11)
confint(lr11)

lr12 <- lm(mwt1best ~ gender, data)
summary(lr12)
confint(lr12)

lr13 <- lm(mwt1best ~ smoking, data)
summary(lr13)
confint(lr13)

lr14 <- lm(mwt1best ~ comorbid, data)
summary(lr14)
confint(lr14)
```
</br>

Variables that are statistically significantly associated with walking distance:
* age
* packhistory
* copdseverity
* fev1
* fev1pred
* fvc
* fvcpred 
* cat
* had
* sgrq
* comorbid
</br>
</br>

Since fev1, fev1pred, fvc and fvcpred are all measures of lung volume, it is expected that they will be highly correlated.
CAT and SGRQ are also expected to be correlated as they are both measures of COPD severity.
this is confirmed by examining the correlation matrix.
</br>

```
#correlation matrix
continuous <- data[, c("age", "packhistory","fev1", "fev1pred", "fvc", "fvcpred", "cat", "had", "sgrq")]
cor_matrix <- cor(continuous, method = "spearman", use = "complete.obs")
cor_matrix
pairs(~age + packhistory + fev1 + fev1pred + fvc + fvcpred + cat + had + sgrq, data = data)
```

</br>
To avoid multicollinearity, variables that are highly correlated must be excluded from the multiple linear regression model.
Amongst the different measures of lung function, fev1 explains the most variance. Therefore it will be included in the final model.
</br> </br>
As CAT, SGRQ and copdseverity are all measures of COPD severity, only one of these will be included in the multiple linear regression model.
Amongst these measures, SGRQ explains the most variance. Therefore it will be included in the final model.
</br> </br>
Packhistory and smoking, age and agequartiles are also likely to be collinear.
Packhistory and age will be favoured for inclusion in the multiple linear regression model as they provide more information.
</br></br>
There are methods of automated variable selection such as stepwise, foward and backwards selection that can help us decide which variables to include in the multiple linear regression model. However, these models 
[come with their limitations](https://www.stata.com/support/faqs/statistics/stepwise-regression-problems/) 
and I prefer to be more intentional about developing a model that best answers the research question.
</br></br>

The predictor variables we will include in the final regression model are:
1. age
2. packhistory
3. fev1
4. had
5. sgrq
6. comorbid

```
# fitting the multiple linear regression model
mlr1<- lm(mwt1best ~ age + packhistory + fev1 + had + sgrq + comorbid, data)
summary(mlr1)
confint(mlr1) 
plot(mlr1)
```
</br>
