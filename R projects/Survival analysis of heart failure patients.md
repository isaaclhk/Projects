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
Whenever there is missing data, it is always important to try and understand why the data is missing, because the reason will tell us the best way to deal with the missing data. For this project, the participants with unknown quintiles were dropped as the proportion of participants in the group is small. Moreover, a cross table between quintile and death would show that there are no deaths among the 4 participants with unknown quantiles. 
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



#Checking linearity assumption
ggcoxfunctional(Surv(fu_time, death)~ los + age + prior_appts_attended + prior_dnas, data = HF)
#linearity assumption not met for prior_appts_attended and prior_dnas. therefore categorize

HF <- HF %>% mutate(
  prior_appts_attended = factor(ifelse(prior_appts_attended <= 10, "<= 10",
                                       ifelse(prior_appts_attended <= 20, "<= 20", "> 20"))),
  prior_dnas = factor(ifelse(prior_dnas == 0, "0", "1")))

describe(HF$prior_appts_attended)
describe(HF$prior_dnas)

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

str(HF)

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

#not unexpectedly large changes in coefficients seen, largest difference seen in metastatic cancer, but expected given the relatively high hazards ratio
#high confidence intervals for metastatic cancer

#testing proportionality assumption

test <- cox.zph(cox_reduced)
print(test)
plot(test)
ggcoxzph(test)
#ihd failed proportionality assumption
#assumption failed for ihd, as confirmed with km plot
km_ihd <- survfit(Surv(fu_time, death)~ ihd, data = HF)
plot(km_ihd, xlab = "overall survival probability", ylab = "time", main = "ihd",
     col = c("red", "blue"))

#strata
cox_reduced2 <- coxph(Surv(fu_time, death)~ 
                        los + age + gender + strata(ihd) + valvular_disease + metastatic_cancer + pneumonia + quintile + ethnicgroup,
                      data = HF)
summary(cox_reduced2)

test2 <- cox.zph(cox_reduced2)
print(test2)
plot(test2)
ggcoxzph(test2)


#outliers
ggcoxdiagnostics(cox_reduced2, type = "deviance",
                 linear.predictions = FALSE, ggtheme = theme_bw())

ggcoxdiagnostics(cox_reduced2, type = "dfbeta",
                 linear.predictions = FALSE, ggtheme = theme_bw())
```
