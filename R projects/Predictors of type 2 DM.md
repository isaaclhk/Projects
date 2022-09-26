# Predictors of Type 2 Diabetes Mellitus

## Background
This analysis was performed on a dataset obtained from the 3rd course of the Statistical Analysis with R for Public Health specialization: Logistic Regression in R for public health. The dataset can be found [here](https://github.com/isaaclhk/Projects/blob/main/datasets/diabetes.csv). The outcome variable for this analysis is dm, a binary variable indicating whether or not a participant is diagnosed with type 2 diabetes mellitus (DM). The aim of this project is to identify predictors of Type 2 DM and quantify their risks.

### Variables in the dataset
1. chol <- total cholesterol in mg/dL
2. stab.glu <- stabilized glucose in mg/dL
3. hdl <- high density lipoprotein in mg/dL
4. ratio <- chol/ hdl
5. glyhb <- Hemoglobin A1C percentage
6. location <- Buckingham or Louisa
7. age <- Age in years
8. gender <- male or female
9. height <- participant's height in inches
10. weight <- participant's weight in pounds
11. frame <- visual estimation of participant's body frame
12. bp.1s <- first measurement of systolic blood pressure
13. bp.1d <- first measurement of diastolic blood pressure
14. bp.2s <- second measurement of systolic blood pressure
15. bp.2d <- second measurement of diastolic blood pressure
16. waist <- circumference of waist in inches
17. hip <- circumference of hips in inches
18. time.ppn <- postprandial time when labs were drawn in minutes
19. insurance <- 0(none), 1(government), 2(private)
20. fh <- family history = 0(no), 1(yes)
21. smoking <- 1(current), 2(never), 3(ex)
22. dm <- yes or no

## Assumptions of logistic regression

1. The response variable is binary
2. The observations are independent- the observations should not be related to each other in any way.
3. There is No Multicollinearity Among Explanatory Variables
4. There are No Extreme Outliers
5. There is a Linear Relationship Between Explanatory Variables and the Logit of the Response Variable

## Code
First, the relevant libraries and dataset were loaded. 
The column names were changed to all lowercase letters for simplicity, then we have a brief overview of the dataset.

```
#loading libraries and dataset
library(tidyverse)
library(Hmisc)
library(crosstable)

diabetes <- read.csv("~/Projects/datasets/diabetes.csv", header= TRUE) %>% rename_all(tolower)
head(diabetes)
str(diabetes)
```

A typical diagnosis criteria for type 2 DM is random blood glucose of >= 200mg/dL or hba1c of >= 6.5%. 
The dataset includes data of each patient's stabilized glucose and hba1c levels.
The data from these variables can be checked against the patient's diagnosis of diabetes to verify the data's accuracy. 

```
#check for patients that meet conventional dm diagnosis criteria but are undiagnosed
diabetes %>% filter(stab.glu >= 200 & dm == "no" | glyhb >= 6.5 & dm == "no")
```

6 patients who were not diagnosed with DM were found to meet the criteria for DM diagnosis. Where possible, this discrepency should be clarified with the researcher.
</br> </br>
After examining the data, 403 observations were noted. Furthermore, variables meant to be categorical were characters or integers. These will be changed to factors.
</br> </br>
It is sometimes useful to categorize patients into groups, for example by BMI or cholesterol levels, as thresholds are often used for clinical decision-making. Categorizing can also make it easier to track proportions of patients who meet certain thresholds. 
</br> </br>
For this analysis, age, BMI and cholesterol were categorised.
stab.glu and glyhb were removed because they are used as diagnostic criteria for diabetes, hence not appropriate as predictors of DM.
time.ppn was also removed as it is not a candidate predictor.

```
#categorizing
diabetes <- diabetes %>% mutate(
  age_group = ifelse(age < 45, "under 45",
                ifelse(age >=45 & age < 65, "45 to 64",
                  ifelse(age >= 65 & age < 75, "65 to 74",
                    ifelse(age >=75, "above 75", NA)))),
  bmi = weight*0.453592 / (height*0.0254)^2,
  bmi_cat = ifelse(bmi <= 18.5, "underweight", 
               ifelse(bmi > 18.5 & bmi <= 25, "normal",
                      ifelse(bmi > 25 & bmi <= 30, "overweight",
                             ifelse(bmi > 30, "obese", NA)))),
  chol_cat = ifelse(chol < 200, "healthy", 
                    ifelse(chol < 240, "borderline high",
                           ifelse(chol >= 240, "high", NA))),
  #converting to factors
  location = factor(location),
  gender = factor(gender),
  frame = factor(frame, ordered = TRUE, levels = c("small", "medium", "large")),
  insurance = factor(insurance),
  fh = factor(fh),
  smoking = factor(smoking), 
  dm = factor(dm),
  age_group = factor(age_group, ordered = TRUE, levels = c("under 45", "45 to 64", "65 to 74", "above 75")),
  bmi_cat = factor(bmi_cat, ordered = TRUE, levels = c("underweight", "normal", "overweight", "obese")),
  chol_cat = factor(chol_cat, ordered = TRUE, levels = c("healthy", "borderline high", "high"))) %>%
  #removing irrelevant variables
  select(-c(x, id, stab.glu, glyhb, time.ppn))

str(diabetes)
```
Each variable was examined individually to observe the data's distribution and identify outliers.

```
#examining individual variables
summary(diabetes$chol, exclude = NULL)
hist(diabetes$chol, main = "chol", breaks = 15)

summary(diabetes$hdl, exclude = NULL)
hist(diabetes$hdl, main = "hdl", breaks = 15)

summary(diabetes$ratio, exclude = NULL)
hist(diabetes$ratio, main = "ratio", breaks = 15) 

summary(diabetes$age, exclude = NULL)
hist(diabetes$age, main = "age", breaks = 15)

summary(diabetes$height, exclude = NULL)
hist(diabetes$height, main = "height", breaks = 15)

summary(diabetes$weight, exclude = NULL)
hist(diabetes$weight, main = "weight", breaks = 15)

summary(diabetes$bp.1s, exclude = NULL)
hist(diabetes$bp.1s, main = "systolic bp 1", breaks = 15)

summary(diabetes$bp.2s, exclude = NULL)
hist(diabetes$bp.2s, main = "systolic bp 2", breaks = 15)

summary(diabetes$bp.1d, exclude = NULL)
hist(diabetes$bp.1d, main = "diastolic bp 1", breaks = 15)

summary(diabetes$bp.2d, exclude = NULL)
hist(diabetes$bp.2d, main = "diastolic bp 2", breaks = 15)

## notice many missing values in 2s and 2d

summary(diabetes$waist, exclude = NULL)
hist(diabetes$waist, main = "waist circumference, inches", breaks = 15)

summary(diabetes$hip, exclude = NULL)
hist(diabetes$hip, main = "hip circumference, inches", breaks = 15)


describe(diabetes$location, exclude.missing = FALSE)
describe(diabetes$gender, exclude.missing = FALSE)
describe(diabetes$frame, exclude.missing = FALSE)
describe(diabetes$insurance, exclude.missing = FALSE)
describe(diabetes$fh, exclude.missing = FALSE)
describe(diabetes$smoking, exclude.missing = FALSE)
describe(diabetes$dm, exclude.missing = FALSE)
describe(diabetes$age_group, exclude.missing = FALSE)
describe(diabetes$bmi_cat, exclude.missing = FALSE) 
describe(diabetes$chol_cat, exclude.missing = FALSE)
```

An outlier was seen on the histogram of the ratio variable. When in doubt, always clarify with the researcher whether or not the data could be spurious.
There were also large amounts of missing data in bp.2s and bp.2d.
Also note that there were only 9 underweight participants. As the distribution of patients in this category was too narrow, the underweight and normal BMI categories were combined to form a single category.

```
#combining BMI categories
levels(diabetes$bmi_cat)
diabetes <- diabetes %>% mutate(bmi_cat = fct_recode(bmi_cat, 
                   "normal or less" = "normal",
                   "normal or less" = "underweight"))
levels(diabetes$bmi_cat)
describe(diabetes$bmi_cat, exclude.missing = FALSE)
```

The relationships between each candidate predictor variable and the outcome variable (DM diagnosis) is analysed. The data is visualized on tables for categorical predictors and plots for continunous predictors. Inspecting the data visually helps to determine whether a linear relationship exists and whether it makes sense to include any particular variable in the final logistic regression model. </br> </br>
Unlike linear regressions, the outcome variable in logistic regressions are binomial and consists of only two values. It doesnt make sense to plot the two values of the outcome variable on the y axis, so we plot the log of their odds instead (also known as logit), which can take on any value from negative to positive infinity. 

```
#examining relationships between individual categorical variables and dm
crosstable(diabetes, location, by = dm) %>% as_flextable()
crosstable(diabetes, gender, by = dm) %>% as_flextable()
crosstable(diabetes, frame, by = dm) %>% as_flextable()
crosstable(diabetes, insurance, by = dm) %>% as_flextable()
crosstable(diabetes, fh, by = dm) %>% as_flextable()
crosstable(diabetes, age_group, by = dm) %>% as_flextable() 
crosstable(diabetes, bmi_cat, by = dm) %>% as_flextable() 
crosstable(diabetes, chol_cat, by = dm) %>% as_flextable()

#examining relationships between individual continous variables and log odds of dm
chol_table <- table(diabetes$chol, diabetes$dm)
prop_chol_table <- prop.table(chol_table, margin = 1)
odds_chol <- prop_chol_table[, 2]/ prop_chol_table[, 1]
logodds_chol <- log(odds_chol)
plot(rownames(chol_table), logodds_chol, main = "cholesterol", xlim = c(140, 300))

hdl_table <- table(diabetes$hdl, diabetes$dm)
prop_hdl_table <- prop.table(hdl_table, margin = 1)
odds_hdl <- prop_hdl_table[, 2]/ prop_hdl_table[, 1]
logodds_hdl <- log(odds_hdl)
plot(rownames(hdl_table), logodds_hdl, main = "hdl", xlim = c(20, 90))

ratio_table <- table(diabetes$ratio, diabetes$dm)
prop_ratio_table <- prop.table(ratio_table, margin = 1)
odds_ratio <- prop_ratio_table[, 2]/ prop_ratio_table[, 1]
logodds_ratio <- log(odds_ratio)
plot(rownames(ratio_table), logodds_ratio, main = "ratio", xlim = c(2, 10))

age_table<- table(diabetes$age, diabetes$dm)
prop_age_table <- prop.table(age_table, margin = 1)
odds_age <- prop_age_table[, 2]/ prop_age_table[, 1]
logodds_age <- log(odds_age)
plot(rownames(age_table), logodds_age, main = "age")

height_table <- table(diabetes$height, diabetes$dm)
prop_height_table <- prop.table(height_table, margin = 1)
odds_height <- prop_height_table[, 2]/ prop_height_table[, 1]
logodds_height <- log(odds_height)
plot(rownames(height_table), logodds_height, main = "height", xlim = c(55, 80), ylim = c(-2.5, -0.5))

weight_table <- table(diabetes$weight, diabetes$dm)
prop_weight_table <- prop.table(weight_table, margin = 1)
odds_weight <- prop_weight_table[,2]/ prop_weight_table[,1]
logodds_weight <- log(odds_weight)
plot(rownames(weight_table), logodds_weight, main = "weight", xlim = c(110, 250))

bp.1s_table <- table(diabetes$bp.1s, diabetes$dm)
prop_bp.1s_table <- prop.table(bp.1s_table, margin = 1)
odds_prop_bp.1s <- prop_bp.1s_table[,2]/ prop_bp.1s_table[1]
logodds_prop_bp.1s <- log(odds_prop_bp.1s)
plot(rownames(bp.1s_table), logodds_prop_bp.1s, main = "systolic bp 1", xlim = c(100, 210))

bp.2s_table <- table(diabetes$bp.2s, diabetes$dm)
prop_bp.2s_table <- prop.table(bp.2s_table, margin = 1)
odds_prop_bp.2s <- prop_bp.2s_table[,2]/ prop_bp.2s_table[1]
logodds_prop_bp.2s <- log(odds_prop_bp.2s)
plot(rownames(bp.2s_table), logodds_prop_bp.2s, main = "systolic bp 2", xlim = c(120, 220))

bp.1d_table <- table(diabetes$bp.1d, diabetes$dm)
prop_bp.1d_table <- prop.table(bp.1d_table, margin = 1)
odds_prop_bp.1d <- prop_bp.1d_table[,2]/ prop_bp.1d_table[1]
logodds_prop_bp.1d <- log(odds_prop_bp.1d)
plot(rownames(bp.1d_table), logodds_prop_bp.1d, main = "diastolic bp 1")

bp.2d_table <- table(diabetes$bp.2d, diabetes$dm)
prop_bp.2d_table <- prop.table(bp.2d_table, margin = 1)
odds_prop_bp.2d <- prop_bp.2d_table[,2]/ prop_bp.2d_table[1]
logodds_prop_bp.2d <- log(odds_prop_bp.2d)
plot(rownames(bp.2d_table), logodds_prop_bp.2d, main = "diastolic bp 2", xlim = c(70, 115))

waist_table <- table(diabetes$waist, diabetes$dm)
prop_waist_table <- prop.table(waist_table, margin = 1)
odds_prop_waist <- prop_waist_table[,2]/ prop_waist_table[1]
logodds_prop_waist <- log(odds_prop_waist)
plot(rownames(waist_table), logodds_prop_waist, main = "waist circumference in inches")

hip_table <- table(diabetes$hip, diabetes$dm)
prop_hip_table <- prop.table(hip_table, margin = 1)
odds_prop_hip <- prop_hip_table[,2]/ prop_hip_table[1]
logodds_prop_hip <- log(odds_prop_hip)
plot(rownames(hip_table), logodds_prop_hip, main = "hip circumference in inches", xlim = c(35, 65))

```

Positive trends between frame, bmi_cat, chol_cat and prevalence of DM were identified.
We also note that the relationship between hdl and log odds of DM diagnosis does not appear linear.

### Variable Selection

When selecting the variables to include in a logistic regression model, it is always important to first review the literature. independent variables that are known to predict the outcome variable should be included in the model regardless of whether their associated p values fall within the threshold.
</br> </br>
Secondly, examining the data as we have done will help us to identify variables that are not suitable for the model. Factors to consider when deciding whether it is appropriate to include a variable in the model include:
1. The proportion of missing data
2. Data with narrow distributions
3. Variables that are collinear with other candidate predictor variables
4. Variables that do not exhibit a linear relationship with the log odds of the outcome variable
</br>

bp.2s and bp.2d were not included in the model as they had large proportions of missing data. There was a narrow distribution of patients in the underweight category of bmi_cat, but that was resolved by combining the normal and underweight levels. hdl was also excluded from the model as it did not appear to have a linear relationship with the log odds of DM diagnosis.

To verify that predictor variables are not collinear, the continuous variables were screened for correlation by forming a correlation matrix. Nominal variables can also be tested for correlation with chi-square test.

```
continuous <-diabetes[, c("chol", "hdl", "ratio", "age", "height", "weight", "bp.1s", "bp.1d", "waist", "hip", "bmi")]
cor(continuous, method = "spearman", use = "complete.obs")
pairs(~chol + hdl + ratio + age + height + weight + bp.1s + bp.1d + waist + hip + bmi, data = diabetes)
```

Based on the matrix, systolic and diastolic blood pressure were expectedly correlated. bmi, waist, hip and weight were also strongly correlated.
chol and ratio were moderately correlated, while hdl and ratio were strongly correlated.
</br> </br>
Between bmi, waist, hip and weight, bmi was chosen as a predictor as the evidence for it is strong in the literature. Between chol and ratio, chol will be chosen as there was an outlying value in ratio. Models that include bp.1s and bp.1d will be tested to identify the better predictor.

```
null_model<- glm(data = diabetes, dm~1, family = binomial(link = "logit"))

model <- glm(data= diabetes, dm~ age + gender + height +location + smoking + fh + bp.1s + bmi + chol + insurance, family = binomial(link = "logit"))
sum_model <- summary(model)
sum_model
exp(sum_model$coefficients)
exp(confint(model))

anova(model, test = "Chisq")
```
The results of the initial model are shown below:

```
Call:
glm(formula = dm ~ age + gender + height + location + smoking + 
    fh + bp.1s + bmi + chol + insurance, family = binomial(link = "logit"), 
    data = diabetes)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.9065  -0.5524  -0.3428  -0.1661   2.7004  

Coefficients:
                 Estimate Std. Error z value Pr(>|z|)    
(Intercept)    -17.014550   4.696964  -3.622 0.000292 ***
age              0.055583   0.012404   4.481 7.43e-06 ***
gendermale      -0.183488   0.465982  -0.394 0.693754    
height           0.109515   0.062915   1.741 0.081740 .  
locationLouisa  -0.160292   0.329095  -0.487 0.626209    
smoking2        -0.083205   0.370928  -0.224 0.822511    
smoking3        -0.122326   0.506242  -0.242 0.809063    
fh1              1.134007   0.365372   3.104 0.001911 ** 
bp.1s            0.006375   0.007432   0.858 0.391010    
bmi              0.078549   0.025357   3.098 0.001950 ** 
chol             0.009968   0.003488   2.858 0.004262 ** 
insurance1      -0.283288   0.388414  -0.729 0.465789    
insurance2      -0.578067   0.401836  -1.439 0.150274    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 324.38  on 378  degrees of freedom
Residual deviance: 255.96  on 366  degrees of freedom
  (24 observations deleted due to missingness)
AIC: 281.96

Number of Fisher Scoring iterations: 6

> exp(sum_model$coefficients)
                   Estimate Std. Error     z value Pr(>|z|)
(Intercept)    4.080140e-08 109.613922  0.02671696 1.000292
age            1.057157e+00   1.012481 88.33102951 1.000007
gendermale     8.323621e-01   1.593578  0.67451187 2.001214
height         1.115736e+00   1.064936  5.70121476 1.085173
locationLouisa 8.518950e-01   1.389710  0.61442460 1.870507
smoking2       9.201621e-01   1.449078  0.79906179 2.276207
smoking3       8.848598e-01   1.659045  0.78534228 2.245802
fh1            3.108086e+00   1.441051 22.28028451 1.001913
bp.1s          1.006396e+00   1.007460  2.35793692 1.478473
bmi            1.081716e+00   1.025681 22.14729993 1.001952
chol           1.010018e+00   1.003494 17.42865953 1.004271
insurance1     7.533025e-01   1.474640  0.48222362 1.593271
insurance2     5.609816e-01   1.494567  0.23726828 1.162153
> exp(confint(model))
Waiting for profiling to be done...
                      2.5 %       97.5 %
(Intercept)    2.718202e-12 0.0002848735
age            1.032578e+00 1.0842709197
gendermale     3.308809e-01 2.0653166214
height         9.894877e-01 1.2666464327
locationLouisa 4.450972e-01 1.6258215395
smoking2       4.480238e-01 1.9320984912
smoking3       3.174662e-01 2.3476317654
fh1            1.509314e+00 6.3696747781
bp.1s          9.916247e-01 1.0211652105
bmi            1.029682e+00 1.1379428558
chol           1.003234e+00 1.0171431111
insurance1     3.494859e-01 1.6150672516
insurance2     2.517428e-01 1.2267936673
```

Age, family history, bmi, and cholesterol were significantly correlated with the log odds of DM diagnosis. It is surprising that blood pressure was not a significant result because the evidence of its relationship with DM is strong in the literature. The exponentiated coefficients show the odds of being diagnosed with DM for every unit change in the corresponding predictor variable. For example, every year increase in age was estimated to multiply the odds of being diagnosed by 1.03, and having family history of DM was estimated to multiply the odds by 1.51. the 95% confidence intervals were also printed.

To understand why the relationship between blood pressure and DM was not significant in this model, the correlations between blood pressure amd other predictor variables in the model was checked for colinearity.

```
#colinearity check
cor.test(diabetes$bp.1s, diabetes$age)
cor.test(diabetes$bp.1s, diabetes$height)
cor.test(diabetes$bp.1s, diabetes$bmi)
cor.test(diabetes$bp.1s, diabetes$chol)

cor.test(diabetes$bp.1d, diabetes$age)
cor.test(diabetes$bp.1d, diabetes$height)
cor.test(diabetes$bp.1d, diabetes$bmi)
cor.test(diabetes$bp.1d, diabetes$chol)

library(car)
vif(model)
```

Based on our investigation, bp.1s was moderately and significantly correlated to age, and mildly but significantly correlated with bmi and cholesterol.
bp.1d is also mildly but significantly correlated with bmi and cholesterol. However, vif values are within acceptable range. </br> </br>

Due to the small sample size and relatively small proportion of participants who are diagnosed with diabetes (n = 60), the number of predictors we can expect to have is few.  A general rule of thumb to avoid over fitting is to have at least 10 events or participants per variable. Therefore, continuous cholesterol and bmi were selected as predictors instead of their categorical counterparts, whose levels are each considered a parameter. Furthermore, backward elimination was applied, retaining variables that have demonstrated a significant relationship with the outcome. A chi-square test helps to inform us which variables to eliminate. Although systolic blood pressure was insignificant, it was retained because of its known relationship with DM.

```
model2 <- glm(data= diabetes, dm~ age + bp.1s + fh + bmi + chol, family = binomial(link = "logit"))
sum_model2 <- summary(model2)
sum_model2
exp(sum_model2$coefficients)
exp(confint(model2))

vif(model2)
```

The model summary of the second model are printed below:

```
Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.7216  -0.5630  -0.3584  -0.2069   2.7581  

Coefficients:
             Estimate Std. Error z value Pr(>|z|)    
(Intercept) -9.335135   1.392694  -6.703 2.04e-11 ***
age          0.048777   0.011267   4.329 1.50e-05 ***
bp.1s        0.005392   0.006959   0.775  0.43848    
fh1          1.096957   0.354464   3.095  0.00197 ** 
bmi          0.065334   0.023185   2.818  0.00483 ** 
chol         0.009756   0.003403   2.867  0.00414 ** 
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 324.38  on 378  degrees of freedom
Residual deviance: 263.76  on 373  degrees of freedom
  (24 observations deleted due to missingness)
AIC: 275.76

Number of Fisher Scoring iterations: 5
```

The standard errors of every variable were reduced, but no unexpectedly large differences were seen (this is a good sign). 
The relationship between bp.1s and DM remained insignificant. </br> </br>
In the third model, bp.1s was replaced with bp.1d.

```
model3 <- glm(data = diabetes, dm~age + bp.1d + fh + bmi + chol, family = binomial(link = "logit"))
sum_model3 <- summary(model3)
sum_model3
exp(sum_model$coefficients)
exp(confint(model))

vif(model3)
```
The model summary of the third model is printed below:

```

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.5961  -0.5674  -0.3514  -0.2048   2.7781  

Coefficients:
             Estimate Std. Error z value Pr(>|z|)    
(Intercept) -9.235788   1.567238  -5.893 3.79e-09 ***
age          0.052363   0.010447   5.012 5.38e-07 ***
bp.1d        0.004838   0.012495   0.387  0.69858    
fh1          1.090801   0.353830   3.083  0.00205 ** 
bmi          0.066542   0.023111   2.879  0.00399 ** 
chol         0.009903   0.003406   2.907  0.00364 ** 
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 324.38  on 378  degrees of freedom
Residual deviance: 264.20  on 373  degrees of freedom
  (24 observations deleted due to missingness)
AIC: 276.2

Number of Fisher Scoring iterations: 5
```
In the third, the relationship between diastolic blood pressure and DM diagnosis is insignificant. </br> </br>

### Assessing model fit and predictive power

The quality of a model can be assess in two ways:
1. Predictive power
2. Goodness of fit </br> </br>

Where possible, it is often advisable to split the dataset into a training set and a test set. Running our model on both sets will help us to verify that the generated model is robust and reproducible. However, this is not possible for this project because of the small sample size.

**Predictive power** </br>
The predictive power of a model can be measured either by using R squared or c statistic. </br>
R squared is also used for assessing linear regression models. in logistic regressions, a similar method (mcfadden's pseudo R squared) with the same interpretation is used. R squared measures the proportion of variance that can be explained by the predictor variables. </br></br>
The c statistic is a measure of discrimination- it measures how well a model can distinguish between those who have and do not have the outcome of interest.
The receiver operating characteristic (ROC) curve is a plot of sensitivity/ 1 - specificity, and the concordonce statistic is the area under the ROC curve. A c statistic of 0.5 indicates that a model is only as good at predicting the outcome as random chance, and a c statistic of 1 indicates perfect prediction.

```
#Calculating McFadden's Pseudo R^2
R2 <-1 - logLik(model)/logLik(null_model)
R2

R2_2 <- 1- loglik(model2)/loglik(null_model)
R2_2

R2_3 <- 1 - loglik(model3)/loglik(null_model)
R2_3

#Plotting ROC curve and calculating C statistic
predicted <- predict(model, diabetes, type = "response")
predicted

predicted2 <- predict(model2, diabetes, type = "response")
predicted2

predicted3 <- predict(model3, diabetes, type = "response")
predicted3

cstat <- round(auc(diabetes$dm, predicted), digits = 4)
cstat

cstat2 <- round(auc(diabetes$dm, predicted2), digits = 4)
cstat2

cstat3 <- round(auc(diabetes$dm, predicted3), digits = 4)
cstat3

roc1 <- plot.roc(diabetes$dm, predicted, col = "red", main = "ROC comparison")
roc2 <- lines.roc(diabetes$dm, predicted2, col = "blue")
roc3 <- lines.roc(diabetes$dm, predicted3, col = "green")
```
Based on the analysis above, model 1 had the best predictive power with an R^2 value of 0.236, followed by model 2 with R^2 = 0.228, and model 3 R^2 = 0.225.
The cstat values also suggest that model 1 had the best predictive power (cstat = 0.819). Cstat for model 2 was 0.808, and 0.805 for model 3. A plot of the ROC curve is shown in the image below. The red line represents model 1, blue = model 2, green = model 3.
</br> ![DM_ROC](https://user-images.githubusercontent.com/71438259/191149558-042135c6-099c-44bd-bdfb-f0149685110f.jpeg)
</br></br>

**Goodness of fit** </br>
The goodness of fit of a model can be measured by examining the residual deviance. The residual deviance is provided in the model summary and is a measure of the difference between the log odds of outcomes in the saturated and the proposed models. To test whether a parameter in the model decreases the deviance by a significant amount for the degrees of freedom taken by the parameter, we can use a chi-square test which generates a p-value. The variables that were eliminated from model 1 were chosen based on the result of the chi-square test.

```
anova(model, test = "Chisq")
anova(model2, test = "Chisq")
anova(model3, test = "Chisq")
```

Another way to assess goodness of fit is to use the Akaike Information Criterion (AIC) provided in the model summary. The AIC aims to describe how well the model fits the data while penalising models with lots of coefficients. In this context, it is the residual deviance adjusted for the number of parameters in the model. AIC is no use by itself but it can be used to compare to or more models, where a smaller value suggests better fit. As shown in the model summaries above, the AIC values for our 3 models were 281.96 (model 1), 275.76 (model 2) and 276.2 (model 3) respectively. This suggests that model 2 has the best fit out of the three.  </br>


A third method to measure goodness of fit is the hosmer-lemeshow statistic and test.
Using this method, participants are grouped into typically 10 groups, according to their predicted values. The observed number of outcomes are compared with the predicted number of outcomes using pearson's chi-square test. A large p value indicates that the model is a good fit.
The hosmer lemeshow method has limitations when sample sizes are too small or too large, and there is no good way of deciding on the number of groups to split the participants into. However, a useful plot of the observed against the expected can be generated from the hosmer-lemeshow test in R.

```
library(ResourceSelection)
HL <- hoslem.test(x = model$y, y = fitted(model), g = 10)
HL

HL2 <- hoslem.test(x = model2$y, y = fitted(model2), g = 10)
HL2

HL3 <- hoslem.test(x = model3$y, y = fitted(model3), g = 10)
HL3


library(ResourceSelection)
HL <- hoslem.test(x = model$y, y = fitted(model), g = 10)
HL

# plot of predicted value's prevalence against observed value's prevalence
plot(x = HL$observed[, "y1"]/ (HL$observed[,"y1"] + HL$observed[, "y0"]),
     y = HL$expected[, "yhat1"]/ (HL$expected[, "yhat1"] + HL$expected[, "yhat0"]))

plot(x = HL2$observed[, "y1"]/ (HL2$observed[,"y1"] + HL2$observed[, "y0"]),
     y = HL2$expected[, "yhat1"]/ (HL2$expected[, "yhat1"] + HL2$expected[, "yhat0"]))

plot(x = HL3$observed[, "y1"]/ (HL3$observed[,"y1"] + HL3$observed[, "y0"]),
     y = HL3$expected[, "yhat1"]/ (HL3$expected[, "yhat1"] + HL3$expected[, "yhat0"]))
```
The hosmer-lemeshow tests indicate that all the 3 models' predicted values are a good match for their observed value.
</br> </br>
Finally, the models are plotted to visualize the predicted probability of having DM and the observed cases.

```
#plot of logistic regression model
plot_data <- data.frame(probability = predicted, dm = diabetes$dm)
plot_data <- plot_data %>% 
  arrange(probability) %>%
  mutate(rank = row_number()) %>%
  na.omit()

plot_data2 <- data.frame(probability = predicted2, dm = diabetes$dm)
plot_data2 <- plot_data %>% 
  arrange(probability) %>%
  mutate(rank = row_number()) %>%
  na.omit()

plot_data3 <- data.frame(probability = predicted3, dm = diabetes$dm)
plot_data3 <- plot_data %>% 
  arrange(probability) %>%
  mutate(rank = row_number()) %>%
  na.omit()

plot1 <- ggplot(data = plot_data, aes(rank, probability)) + 
  geom_point(aes(color = dm), alpha = 0.7, stroke = 2, shape = 4, size = 0.4) +
  theme_bw() + 
  labs(x = "Index", y = "Predicted probability of having DM", title = "model 1")

plot2 <- ggplot(data = plot_data2, aes(rank, probability)) + 
  geom_point(aes(color = dm), alpha = 0.7, stroke = 2, shape = 4, size = 0.4) +
  theme_bw() +
  labs(x = "Index", y = "Predicted probability of having DM", title = "model 2")

plot3 <- ggplot(data = plot_data3, aes(rank, probability)) + 
  geom_point(aes(color = dm), alpha = 0.7, stroke = 2, shape = 4, size = 0.4) +
  theme_bw() +
  labs(x = "Index", y = "Predicted probability of having DM", title = "model 3")

plot1
plot2
plot3
```
![DM_lrm_plot](https://user-images.githubusercontent.com/71438259/191163940-6410b658-cac8-4138-9bf6-4bf6aafc2490.jpeg)

## Conclusion

Out of the 3 models, model 1 has the best predictive power. This is expected because it has the most number of predictors. However, having too many predictors when the sample size is small can lead to overfitting which makes the model less robust. Between models 2 and 3, model 2 has slightly better fit and predictive power.
</br> </br>
The results of this analysis show that older age, family history, higher bmi and higher cholesterol increase the risk of having DM. Although  systolic blood pressure was found to be a slightly better predictor of DM than diastolic blood pressure, both blood pressure variables were not significant in our models despite the strong evidence of its relationship with DM in the literature. This insignificance may reasonably be imputed to insufficient sample size and some collinearity between blood pressure and other variables in the model.







