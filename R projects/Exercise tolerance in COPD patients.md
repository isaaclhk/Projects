# Predictors of Exercise Tolerance in Patients with COPD
## Background
This analysis performed on a dataset obtained from the 2nd course of the Statistical Analysis with R for Public Health specialization: Linear Regression in R for public health.
The dataset can be found [here](https://github.com/isaaclhk/Projects/blob/main/datasets/COPD.csv).
The outcome variable for this analysis is MWT1Best, a measure of walking distance, which is used as an indicator of exercise tolerance.
The aim is to identify factors that predict exercise tolerance in patients with COPD and produce a regression model that will best quantify the impact of these factors.

### Variables in the dataset
1. Age <- Age in years
2. Packhistory <- number of packs of cigarettes smoked per day * number of smoking years
3. COPD severity <- levels = c(mild, moderate, severe, very severe)
4. MWT1 <- Distance covered over first attempt of 6-minute walking test
5. MWT2 <- Distance covered over second attempt of 6- minute walking test
6. MWT1Best <- Best performance between MWT1 and MWT2
7. FEV1 <- forced expiratory volume for the first second of exhalation
8. FEV1PRED <- percentage of FEV1 relative to predicted value
9. FVC <- forced vital capacity (total volume of air exhaled during FEV)
10. FVCPRED <- percentage of FVC relative to predicted value
11. CAT <- COPD Assessment Test, a questionnaire to quantify the impact of COPD on a person's life
12. HAD <- Hospital Anxiety and Depression Scale, a questionnaire to measure anxiety and depression
13. SGRQ <- St George's Respiratory Questionnaire, measures impact on daily life and perceived well-being in patients with obstructive airways disease.
14. AGEquartiles <- Age variable categorised into quartiles
15. COPD <- Indication of COPD severity, 1:4 (mild to very severe)
16. Gender <- 0(female), 1(male)
17. Smoking <- 0(non-smokers), 1(ex-smokers), 2(current smokers)
18. Other comorbidities e.g. Diabetes, muscular, hypertension, AtrialFib, IHD- levels - 0(absent), 1(present)

Continuous variables: age, packhistory, mwt1, mwt2, mwt1best, fev1, fev1pred, fvc, fvcpred, cat, had, sgrq </br>
Categorical variables: copdseverity, agequartiles, copd, smoking </br>
Binary variables: gender, diabetes, muscular, hypertension, atrialfib, ihd

## Code
First, the relevant libraries and dataset were loaded. The column names were changed to all lowercase letters for simplicity, 
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

1. The sample size is n = 101.
2. Missing values are present in mwt1, mwt2, and mwt1best.
3. Most variables are numeric, except for copdseverity which is character. Categorical variables will need to be changed to factors.

Missing values are a common problem. In a research study, we can always approach the researcher to ask them about to the missing value. 
However, since this dataset is fictitious, the missing value will be left empty. </br> </br>

The categorical and binary variables will be changed to factors, and the first two columns of the dataset were removed as they were not required.
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
Each continuous variable was examined individually. This gives us a sense of the variable's distribution and help us to identify outliers.
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
However for this project, the incorrect data was removed.
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
As with the continuous variables, the categorical and binary variables were also examined.
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
Hence, copdseverity was removed. mwt1 and mwt2 were also removed as they are measures that directly determine the outcome mwt1best, so they should not be considered as candidate predictors.
</br>

```
data <- data %>% select(-c(copd, mwt1, mwt2))
```

</br>
Due to limitations of sample size, comorbid conditions could not be analysed as individual predictors. 
They were combined to a single binomial variable "comorbid" to reduce the number of predictors and avoid overfitting.
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
The relationships between each candidate predictor variable and the outcome variable were analysed individually by visualizing them on plots and fitting simple linear regression models. The least-squares regression lines were also added to the plots for clarity.
By visualizing the data as such, we can determine whether linear relationships exist between the independent and outcome variables, which informs us whether or not it makes sense to include any particular variable in the multiple linear regression model. 
</br>

```
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

### Variable Selection

When selecting the variables to include in a multiple linear regression model, it is always important to first review the literature. 
independent variables that are known to predict the outcome variable should be included in the model regardless of whether their associated p values fall within the threshold. </br> </br>

Secondly, examining the data as we have done will help us to identify variables that are not suitable for the model. Factors to consider when deciding whether it is appropriate to include a variable in the model include:
1. The proportion of missing data, 
2. Data with narrow distributions 
3. Variables that are collinear with other candidate predictor variables
4. Variables that do not exhibit a linear relationship with the outcome variable
</br> </br>

To avoid multicollinearity, it is important to ensure that the selected predictors are independent.
Since fev1, fev1pred, fvc and fvcpred are all measures of lung volume, it is expected for them to be highly correlated.
CAT and SGRQ are also expected to be correlated as they are both measures of COPD severity.
This was confirmed by examining the correlation matrix.
</br>

```
#correlation matrix
continuous <- data[, c("age", "packhistory","fev1", "fev1pred", "fvc", "fvcpred", "cat", "had", "sgrq")]
cor_matrix <- cor(continuous, method = "spearman", use = "complete.obs")
cor_matrix
pairs(~age + packhistory + fev1 + fev1pred + fvc + fvcpred + cat + had + sgrq, data = data)
```

</br>
variables that are highly correlated must be excluded to ensure indepdence of predictor variables in the multiple linear regression model.
Amongst the different measures of lung function, fev1 explains the most variance. Therefore it was included in the final model.
</br> </br>
As CAT, SGRQ and copdseverity are all measures of COPD severity, only one of these were included in the multiple linear regression model.
Amongst these measures, SGRQ explains the most variance. Therefore it was be included in the final model.
</br> </br>
Packhistory and smoking, age and agequartiles are also likely to be collinear.
Packhistory and age were favoured for inclusion in the multiple linear regression model as they provide more information.
</br></br>
There are methods of automated variable selection such as stepwise, foward and backwards selection that can help us decide which variables to include in the multiple linear regression model. </br>

However, such methods [come with limitations](https://www.stata.com/support/faqs/statistics/stepwise-regression-problems/).
It is imperative to be intentional about developing a model that best answers the research question by manually selecting predictors. However, when elimination is necessary, backwards selection can still be used with caution.
</br></br>

The predictor variables included in the final regression model were:
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

#checking for collinearity
imcdiag(mod = mlr1, method = "VIF")
```
</br>

## Results

Based on the plots, the residuals are somewhat homogeneous and normally distributed.
![COPD_homoscedasticity](https://user-images.githubusercontent.com/71438259/189305048-0bb9cf13-dff0-49da-a485-dae53dbb3f2f.jpeg)

![COPD_QQplot](https://user-images.githubusercontent.com/71438259/189305176-a11383e6-5c22-44a6-856b-fbf16b05e716.jpeg)

A calculation of variance inflation factors also show that multicollinearity is not a problem.
</br>

```
> imcdiag(mod = mlr1, method = "VIF")

Call:
imcdiag(mod = mlr1, method = "VIF")


 VIF Multicollinearity Diagnostics

               VIF detection
age         1.1720         0
packhistory 1.1150         0
fev1        1.1825         0
had         1.5606         0
sgrq        1.5108         0
comorbid1   1.2500         0

NOTE:  VIF Method Failed to detect multicollinearity


0 --> COLLINEARITY is not detected by the test
```
</br>

The results of the final model are printed.

```
> summary(mlr1)

Call:
lm(formula = mwt1best ~ age + packhistory + fev1 + had + sgrq + 
    comorbid, data = data)

Residuals:
    Min      1Q  Median      3Q     Max 
-150.30  -52.37   -3.26   56.11  174.38 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) 828.5206    99.6418   8.315 1.66e-12 ***
age          -4.5891     1.1922  -3.849 0.000234 ***
packhistory  -0.8465     0.3629  -2.332 0.022133 *  
fev1         31.8759    13.9105   2.292 0.024499 *  
had          -2.3822     1.8347  -1.298 0.197786    
sgrq         -2.3746     0.6149  -3.862 0.000224 ***
comorbid1    -8.4218    18.6878  -0.451 0.653425    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 78.6 on 82 degrees of freedom
  (12 observations deleted due to missingness)
Multiple R-squared:  0.473,	Adjusted R-squared:  0.4344 
F-statistic: 12.26 on 6 and 82 DF,  p-value: 8.396e-10

> confint(mlr1)
                 2.5 %       97.5 %
(Intercept) 630.301269 1026.7399345
age          -6.960759   -2.2174311
packhistory  -1.568551   -0.1245139
fev1          4.203541   59.5482014
had          -6.031927    1.2675951
sgrq         -3.597925   -1.1513415
comorbid1   -45.597693   28.7540925
```

### Model Intepretation

In the final model, age, packhistory, FEV1, and SGRQ are significantly associated with walking tolerance in COPD patients.</br>
This model explains 43.4% of variance in walking distance among COPD patients from the 6- minute walk test. The adjusted R-squared of 0.434 is statistically significant at the conventional threshold p < .05. </br>
The model suggests that walking distance of COPD patients in the 6- minute walk will likely reduce by an average of 4.59 metres for every year the patient is older. 
An increase in 1 unit of packhistory decreases walking distance in the test by an average of about -0.85 metres. Every unit of SGRQ score decreases walking distance in the test by an average 2.37 metres. Having at least 1 comorbid condition decreases walking distance in the test by an average of 8.4 metres. However, there is no strong evidence that having comorbid medical conditions affects walking distance in COPD patients as it's p value of 0.65 is far above the threshold of .05. Furthermore, the confidence interval is too large to give us enough information about it's relationship with walking distance. A 1 unit increase of FEV1 was estimated to improve walking distance in the test by an average of 31.89 metres. The confidence interval for fev1 is also large, but we can be mostly confident that fev1 has a positive relationship with walking distance as the estimate is positive even at alpha = .025.
