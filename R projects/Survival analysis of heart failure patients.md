# Survival analysis of heart failure patients

## Background

This analysis was performed on a dataset obtained from the final course of the Statistical Analysis with R for Public Health specialization: Survival analysis in R for public health. The dataset can be found [here](https://github.com/isaaclhk/Projects/blob/main/datasets/HF_mort_data.csv). This project aims to identify factors that influence risk of mortality in patients with heart failure and estimate their corresponding hazard ratios.

### Variables in the dataset
1. death (0/1)
2. los (hospital length of stay in nights)
3. age (in years)
4. gender (1=male, 2=female)
5. cancer
6. cabg (previous heart bypass)
7. crt (cardiac resynchronisation device - a treatment for heart failure)
8. defib (defibrillator implanted)
9. dementia
10. diabetes (any type)
11. hypertension
12. ihd (ischaemic heart disease)
13. mental_health (any mental illness)
14. arrhythmias
15. copd (chronic obstructive lung disease)
16. obesity
17. pvd (peripheral vascular disease)
18. renal_disease
19. valvular_disease (disease of the heart valves)
20. metastatic_cancer
21. pacemaker
22. pneumonia
23. prior_appts_attended (number of outpatient appointments attended in the previous year)
24. prior_dnas (number of outpatient appointments missed in the previous year)
25. pci (percutaneous coronary intervention)
26. stroke (history of stroke)
27. senile
28. quintile (socio-economic status for patient's neighbourhood, from 1 (most affluent) to 5 (poorest))
29. fu_time (follow-up time, i.e. time in days since admission to hospital) 
30. ethnicgroup has the following categories in this extract: 1=white, 2=black, 3=Indian subcontinent, 8=not known, 9=other

## Assumptions of cox proportional hazards regression

1. non-informative censoring- likelihood of observation being censored is not related to likelihood of the event occuring
2. independent survival times- survival time of each observation is not related to that in any other observation.
3. Hazard ratios are proportional- the ratio of the hazards for any two individuals is constant over time.
4. ln(hazard) is a linear function of the predictor variables
5. values of predictor variables dont change over time

## Code
First, the relevant libraries and dataset were loaded. The column names were changed to all lowercase letters for simplicity, then we have a brief overview of the dataset. 

```
#loading libraries and dataset
HF <- read.csv("~/Projects/datasets/HF_mort_data.csv") %>% rename_all(tolower)

library(tidyverse)
library(Hmisc)
library(survival)
library(survminer)

#examining structure of dataset
head(HF)
str(HF) #all integers
```
There are 1000 observations, and all variables are integers. </br> </br>
Categorical variables are changed to factors. senile and dementia variables were combined to form the variable "cognitive_imp", then id was removed as it is not useful for the analysis.

```
HF <- HF %>% mutate(
  gender = factor(gender),
  cancer = factor(cancer),
  cabg = factor(cabg),
  crt = factor(crt),
  defib = factor(defib),
  cognitive_imp = factor(ifelse(dementia == 1, 1, 
                         ifelse(senile == 1, 1, 0))),
  diabetes = factor(diabetes),
  hypertension = factor(hypertension),
  ihd = factor(ihd),
  mental_health = factor(mental_health),
  arrhythmias = factor(arrhythmias),
  copd = factor(copd),
  obesity = factor(obesity),
  pvd = factor(pvd),
  renal_disease = factor(renal_disease),
  valvular_disease = factor(valvular_disease),
  metastatic_cancer = factor(metastatic_cancer),
  pacemaker = factor(pacemaker),
  pneumonia = factor(pneumonia),
  pci = factor(pci),
  stroke = factor(stroke),
  quintile = factor(quintile),
  ethnicgroup = factor(ethnicgroup)) %>%
select(-c(id, dementia, senile))

str(HF)
```

Each variable was examined individually to observe the data's distribution and identify outliers.

```
#examining variables individually
death_table <- table(HF$death, exclude = NULL)
addmargins(death_table, FUN = sum)
prop.table(death_table)

prior_dnas_table <- table(HF$prior_dnas, exclude = NULL)
addmargins(prior_dnas_table, FUN = sum)
prop.table(prior_dnas_table)

describe(HF$gender)
describe(HF$cancer)
describe(HF$cabg)
describe(HF$crt) 
describe(HF$defib) 
describe(HF$diabetes)
describe(HF$hypertension)
describe(HF$ihd)
describe(HF$mental_health)
describe(HF$arrhythmias)
describe(HF$copd)
describe(HF$obesity)
describe(HF$pvd)
describe(HF$renal_disease)
describe(HF$valvular_disease)
describe(HF$metastatic_cancer)
describe(HF$pacemaker)
describe(HF$pneumonia)
describe(HF$pci)
describe(HF$stroke)
describe(HF$quintile) 
describe(HF$ethnicgroup) 
describe(HF$cognitive_imp)

summary(HF$los)
hist(HF$los, main = "length of stay")

summary(HF$age)
hist(HF$age, main = "age")

summary(HF$prior_appts_attended)
hist(HF$prior_appts_attended, main = "prior appts attended")

summary(HF$prior_dnas)
hist(HF$prior_dnas, main = "prior outpatient appointments missed")

summary(HF$fu_time)
hist(HF$fu_time, main = "follow up time")
```
out of 1000 participants, only 3 had cardiac rechronization devices, and 6 had implanted defibrillators. As there were so few participants with these devices,
"crt", "defib", and "pacemaker" variables were combined to form the "cardiac_device" variable. </br></br>
Furthermore, there are 4 participants with unknown "quintile" values.</br>
Whenever there is missing data, it is always important to try and understand why the data is missing, because the reason will tell us the best way to deal with the missing data. For this project, the participants with unknown quintiles were dropped as the proportion of participants in the group is small and its effect is likely negligible. Moreover, a cross table between quintile and death would show that there are no deaths among the 4 participants with unknown quantiles. 
If this were to be included in a cox regression model, the model would fail to converge. </br></br>
43 participants were found to have missing ethnicgroup values. As this proportion of participants is sizable, a new level of ethnicgroup, "unknown", was added to the variable.</br></br>
More information on the various techniques for handling missing data can be found [in this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3668100/).

```
#combining cardiac device factors, removing level 0 in quintile
HF <- HF %>% mutate(
  cardiac_device = factor(ifelse(crt == 1 | defib == 1 | pacemaker == 1, 1, 0)),
  quintile = fct_recode(quintile, NULL = "0"),
  ethnicgroup = fct_recode(ethnicgroup, 
                           white = "1",
                           black = "2",
                           indian_subcont = "3",
                           others = "9"),
  ethnicgroup = fct_explicit_na(ethnicgroup, na_level = "unknown")) %>%
  select(-c(crt, defib, pacemaker))
  
describe(HF$cardiac_device)
describe(HF$quintile)
```

The linearity assumption was verified for all continuous variables. Although prior_appts_attended and prior_dnas are ordinal variables, they were also checked for linearity because they have many levels. If they turn out to be linear, then they can be treated as continuous for this analysis. Otherwise, we'd have to combine their levels and analyse them as categorical variables.


```
#Checking linearity assumption
ggcoxfunctional(Surv(fu_time, death)~ los + age + prior_appts_attended + prior_dnas, data = HF)
#linearity assumption not met for prior_appts_attended and prior_dnas. therefore categorize!

```

![HF_martingales](https://user-images.githubusercontent.com/71438259/192422466-9e161395-2933-4eb3-9c51-8e68585e76be.jpeg)

As observed from the plots of martingale residuals above, "los" and "age" are somewhat linear, but "prior_appts_attended" and "prior_dnas" have failed to meet the linearity assumption. Therefore, their levels were grouped and analysed as categorical variables. "prior_appts_attended" was factored into 3 levels: <=10, >10 <= 20,
and >20. "prior_dnas" was dichotomised to either 0(no) or 1(yes). The decision for this factorization was based on the distribution of these variables as examined earlier.

```
HF <- HF %>% mutate(
  prior_appts_attended = factor(ifelse(prior_appts_attended <= 10, "<= 10",
                                       ifelse(prior_appts_attended <= 20, "> 10 <= 20", "> 20"))),
  prior_dnas = factor(ifelse(prior_dnas == 0, "0", "1")))

describe(HF$prior_appts_attended)
describe(HF$prior_dnas)
```

The log-rank tests is a great way to perform survival analysis for a single categorical variable. </br>
The advantage of using the log-rank test is that we don’t need to know anything about the shape of the survival curve or the distribution of survival times.
It compares the observed numbers and expected number of events using a chi-square test to determine if there is a difference in probability of event between the groups. </br></br>
kaplan-meier plots help us to visualize the survival probabilities of the variable being analysed over time.

```
#kaplan-meier plots
##no predictors
km_fit <- survfit(Surv(fu_time, death) ~ 1, data = HF)
summary(km_fit)
plot(km_fit)

##prior appointments attended
prior_appts_km <- survfit(Surv(fu_time, death)~ prior_appts_attended, data = HF)
summary(prior_appts_km, times = c(1:10, 30, 60, 90*(1:10)))
plot(prior_appts_km, xlab = "time", ylab = "overall survival probability", main = "prior appointments",
     col = c("red", "darkgreen", "blue"))
survdiff(Surv(fu_time, death)~ prior_appts_attended, data = HF)

##prior appointments missed
prior_dnas_km <- survfit(Surv(fu_time, death)~ prior_dnas, data = HF)
summary(prior_dnas_km, times = c(1:10, 30, 60, 90*(1:10)))
plot(prior_dnas_km, xlab = "time", ylab = "overall survival probability", main = "outpatient appointments missed",
     col = c("red", "blue"))
survdiff(Surv(fu_time, death)~ prior_dnas, data = HF)
```
Cox regression is more useful for analyzing multiple variables at once. First, all variables were included in the cox regression model. The variables that were statistically non-significant at the conventional p value of .05 were then removed by backward elimination, then the remaining variables were analysed in another model.

```
#fitting cox regression
cox <- coxph(Surv(fu_time, death)~ 
    los + age + gender + cancer + cabg + diabetes + hypertension + ihd + 
    mental_health + arrhythmias + copd + obesity + pvd + renal_disease + 
    valvular_disease + metastatic_cancer +pneumonia + prior_appts_attended + 
    prior_dnas + pci + stroke + quintile + ethnicgroup + cognitive_imp + cardiac_device,
    data = HF)
summary(cox)

cox_reduced <- coxph(Surv(fu_time, death)~ 
            los + age + gender + ihd + valvular_disease + metastatic_cancer + pneumonia + quintile + ethnicgroup,
            data = HF)
summary(cox_reduced)
```
After performing backward elimination,  ethnicgroup was no longer statistically significant. The variable will be removed in the next step of elimination according to the pre-determined p-value threshold. No unexpectedly large changes in coefficients were seen after the elimination. The largest difference seen was in metastatic cancer, but this was expected given the relatively high hazards ratio.

```
#remove ethnic group
cox_reduced2 <- coxph(Surv(fu_time, death)~ 
                        los + age + gender + ihd + valvular_disease + metastatic_cancer + pneumonia + quintile,
                      data = HF)
summary(cox_reduced2)
```
No unexpectedly large chnges in coefficents were seen after this step of elimination. before interpreting the results proper, the assumption of proportional hazards was tested.

```
#testing proportionality assumption
test <- cox.zph(cox_reduced2)
print(test)
plot(test)
ggcoxzph(test)
```

Based on the diagnostics given by schoenfeld residuals, "ihd" failed to meet the statistical assumption of proportional hazards (p= 0.0032). The results of this test is printed below. The failed assumption is further confirmed by reviewing the kaplan-meier plot of the "ihd" variable.

```
                   chisq df      p
los                0.145  1 0.7031
age                0.270  1 0.6031
gender             0.702  1 0.4022
ihd                8.694  1 0.0032
valvular_disease   0.363  1 0.5468
metastatic_cancer  0.190  1 0.6626
pneumonia          1.289  1 0.2562
quintile           1.937  4 0.7474
ethnicgroup        0.302  4 0.9897
GLOBAL            14.169 15 0.5128

```
</br>

![HF_KM_ihd](https://user-images.githubusercontent.com/71438259/192427276-f9011e04-dfb2-486b-a2a0-46f55b5b43d1.jpeg)</br></br>

In the kaplan-meier plot above, the lines between ihd and no ihd were crossed at some point, which shows that the hazard ratios between the two groups were not constant over time. </br></br>
One way of dealing with this problem is to stratify the analysis by ihd. This allows us to estimate effects in different strata and then average them together. 
Other methods for addressing non-proportional hazards can be [read here](https://cran.r-project.org/web/packages/Greg/vignettes/timeSplitter.html).

```
#strata
cox_reduced3 <- coxph(Surv(fu_time, death)~ 
                        los + age + gender + strata(ihd) + valvular_disease + metastatic_cancer + pneumonia + quintile + ethnicgroup,
                      data = HF)
summary(cox_reduced3)
```

After stratifying ihd, valvular disease was no longer statistically significant. Therefore, it is eliminated according to the predetermined threshold of p<.05. The proportional hazards assumption was also rechecked.

```
cox_reduced4 <- coxph(Surv(fu_time, death)~ 
                        los + age + gender + strata(ihd) + metastatic_cancer + pneumonia + quintile,
                      data = HF)
                      
#rechecking proportionality assumption
test2 <- cox.zph(cox_reduced4)
print(test2)
plot(test2)
ggcoxzph(test2)
```
The proportional hazards assumption was met in this model.
</br></br>
To test influential observations or outliers, we can visualize either:
1. the deviance residuals 
2. the dfbeta values

Specifying the argument type = “dfbeta”, plots the estimated changes in the regression coefficients upon deleting each observation in turn; likewise, type=“dfbetas” produces the estimated changes in the coefficients divided by their standard errors. 
</br></br>
Specifying the argument type = "deviance", generates a plot of the deviance residuals.
In a normal distribution, 5% of observations are more than 1.96 standard deviations from the mean. So if the SD is 1, then only 5% of observations should be bigger than 1.96 or more negative than -1.96. If more than that is proportion is present, then the model doesn’t fit the data as well as it should and some observations are a problem.
</br>
-Positive values correspond to individuals that “died too soon” compared with expected survival times.
-Negative values correspond to individual that “lived too long” compared with expected survival times.
-Very large or small values are outliers, which are poorly predicted by the model. 

```
#outliers
ggcoxdiagnostics(cox_reduced4, type = "deviance",
                 linear.predictions = FALSE, ggtheme = theme_bw())

ggcoxdiagnostics(cox_reduced4, type = "dfbeta",
                 linear.predictions = FALSE, ggtheme = theme_bw())
```
![HF_devianceresiduals](https://user-images.githubusercontent.com/71438259/192492210-8f0f10ab-58d5-45d3-9f87-de35f3c5bde4.jpeg)

In the plot of deviance residuals above, the pattern looks fairly symmetrical around 0 and the model fits the data reasonably well.

## Model interpretation

The results of the final model are printed below:

```
Call:
coxph(formula = Surv(fu_time, death) ~ los + age + gender + strata(ihd) + 
    metastatic_cancer + pneumonia + quintile, data = HF)

  n= 990, number of events= 489 
   (10 observations deleted due to missingness)

                        coef exp(coef)  se(coef)      z Pr(>|z|)    
los                 0.012507  1.012586  0.003166  3.950 7.81e-05 ***
age                 0.060862  1.062752  0.005552 10.962  < 2e-16 ***
gender2            -0.281254  0.754836  0.096075 -2.927  0.00342 ** 
metastatic_cancer1  2.264963  9.630764  0.371369  6.099 1.07e-09 ***
pneumonia1          0.394225  1.483234  0.138629  2.844  0.00446 ** 
quintile2          -0.379564  0.684160  0.155654 -2.439  0.01475 *  
quintile3           0.067889  1.070246  0.152331  0.446  0.65584    
quintile4          -0.066156  0.935985  0.151918 -0.435  0.66322    
quintile5           0.084749  1.088444  0.150957  0.561  0.57452    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

                   exp(coef) exp(-coef) lower .95 upper .95
los                   1.0126     0.9876    1.0063    1.0189
age                   1.0628     0.9410    1.0513    1.0744
gender2               0.7548     1.3248    0.6253    0.9112
metastatic_cancer1    9.6308     0.1038    4.6510   19.9421
pneumonia1            1.4832     0.6742    1.1303    1.9463
quintile2             0.6842     1.4616    0.5043    0.9282
quintile3             1.0702     0.9344    0.7940    1.4426
quintile4             0.9360     1.0684    0.6950    1.2606
quintile5             1.0884     0.9187    0.8097    1.4632

Concordance= 0.697  (se = 0.012 )
Likelihood ratio test= 205.8  on 9 df,   p=<2e-16
Wald test            = 196.3  on 9 df,   p=<2e-16
Score (logrank) test = 210.1  on 9 df,   p=<2e-16
```

The results of this analysis show that length of stay in the hospital, age, gender, having metastatic cancer, pneumonia, and socioeconomic status of a participants' neighbourhood are factors that influence survival rates of heart failure patients. Patients typically have a 1.2% increase in hazard for each additional day of staying at the hospital. Males generally have a 32.5% higher hazard for mortality compared to females. Those with metastatic cancer have hazards that are on average 9.63 times of those without. The confidence intervals for metastatic cancer is large (4.65 to 19.94), so the prediction is not very precise. However, the result is highly statistically significant and should convince us that metastatic cancer increases mortality rates in patients with heart failure. Patients with pneumonia, on average have 48% higher hazards compared to those without. Lastly, patients who live in regions associated with quintile 2 have 31.6% lower hazards than those in quintile1. When compared with patients from other quintiles, the hazard ratios are not statistically significant.
</br></br>
Concordance is the c-statistic that refers to the area under the curve of receiver operating characteristic curve.
The concordance or c-statistic is a measure of the model's predictive power. A c-statistic of 0.5 indicates that the model is only as good as predicting outcomes as random chance, and a c-statistic of 1 indicates perfect prediction. In this project, the concordance of the final model is 0.697.

