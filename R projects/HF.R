
HF <- read.csv("~/Projects/datasets/HF_mort_data.csv") %>% rename_all(tolower)

library(survival)
library(tidyverse)
library(survminer)
library(Hmisc)

#examining structure of dataset
head(HF)
str(HF) #all integers


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
describe(HF$crt) #only 3 with crt
describe(HF$defib) # only 6 with defib
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
describe(HF$quintile) #4 in level 0, 6 missing
describe(HF$ethnicgroup) #43 missing values
describe(HF$cognitive_imp)


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
  
str(HF)
describe(HF$cardiac_device)
describe(HF$quintile)

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


#Checking linearity assumption
ggcoxfunctional(Surv(fu_time, death)~ los + age + prior_appts_attended + prior_dnas, data = HF)
# linearity assumption not met for prior_appts_attended and prior_dnas. therefore categorize

HF <- HF %>% mutate(
  prior_appts_attended = factor(ifelse(prior_appts_attended <= 10, "<= 10",
                                       ifelse(prior_appts_attended <= 20, "> 10 <= 20", "> 20"))),
  prior_dnas = factor(ifelse(prior_dnas == 0, "0", "1")))

describe(HF$prior_appts_attended)
describe(HF$prior_dnas)

#kaplan-meier plots
##no predictors
km_fit <- survfit(Surv(fu_time, death) ~ 1, data = HF)
summary(km_fit)
plot(km_fit)

## prior appointments attended
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
# high confidence intervals for metastatic cancer

#remove ethnic group
cox_reduced2 <- coxph(Surv(fu_time, death)~ 
                        los + age + gender + ihd + valvular_disease + metastatic_cancer + pneumonia + quintile,
                      data = HF)
summary(cox_reduced2)


#testing proportionality assumption
test <- cox.zph(cox_reduced2)
print(test)
plot(test)
ggcoxzph(test)
#ihd failed proportionality assumption
#assumption failed for ihd, as confirmed with km plot
km_ihd <- survfit(Surv(fu_time, death)~ ihd, data = HF)
plot(km_ihd, xlab = "overall survival probability", ylab = "time", main = "ihd",
     col = c("red", "blue"))

#strata
cox_reduced3 <- coxph(Surv(fu_time, death)~ 
                        los + age + gender + strata(ihd) + valvular_disease + metastatic_cancer + pneumonia + quintile,
                      data = HF)
summary(cox_reduced3)

#remove valvular disease
cox_reduced4 <- coxph(Surv(fu_time, death)~ 
                        los + age + gender + strata(ihd) + metastatic_cancer + pneumonia + quintile,
                      data = HF)
summary(cox_reduced4)

test2 <- cox.zph(cox_reduced4)
print(test2)
plot(test2)
ggcoxzph(test2)


#outliers
ggcoxdiagnostics(cox_reduced4, type = "deviance",
                 linear.predictions = FALSE, ggtheme = theme_bw())

ggcoxdiagnostics(cox_reduced4, type = "dfbeta",
                 linear.predictions = FALSE, ggtheme = theme_bw())

