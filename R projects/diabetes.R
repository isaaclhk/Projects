
#loading libraries and dataset
library(tidyverse)
library(Hmisc)
library(crosstable)

diabetes <- read.csv("~/Projects/datasets/diabetes.csv", header= TRUE) %>% rename_all(tolower)
head(diabetes)
str(diabetes)

#check for patients that meet conventional dm diagnosis criteria but are undiagnosed
diabetes %>% filter(stab.glu >= 200 & dm == "no" | glyhb >= 6.5 & dm == "no")

#adding new variables
# stab.glu and glyhb will be removed because they are used as diagnostic criteria for diabetes, hence not appropriate to be used as a predictor for our outcome DM.
# time.ppn will be removed as its not relevant as a predictor

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
# continuous: chol, stab.glu, hdl, ratio, glyhb, age, height, weight, bp.1s, bp.1d, bp.2s, bp.2d, waist, hip, time.ppn
# categorical: location, gender, frame, insurance, fh, smoking, dm, age_group, bmi_cat, chol_cat

#examining individual variables
summary(diabetes$chol, exclude = NULL)
hist(diabetes$chol, main = "chol", breaks = 15)

summary(diabetes$hdl, exclude = NULL)
hist(diabetes$hdl, main = "hdl", breaks = 15)

summary(diabetes$ratio, exclude = NULL)
hist(diabetes$ratio, main = "ratio", breaks = 15) #outlier found

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
describe(diabetes$bmi_cat, exclude.missing = FALSE) #only 9 underweight
describe(diabetes$chol_cat, exclude.missing = FALSE)

#combining BMI categories
levels(diabetes$bmi_cat)
diabetes <- diabetes %>% mutate(bmi_cat = fct_recode(bmi_cat, 
                   "normal or less" = "normal",
                   "normal or less" = "underweight"))

describe(diabetes$bmi_cat, exclude.missing = FALSE)

#examining relationships between individual categorical variables and dm
crosstable(diabetes, location, by = dm) %>% as_flextable()
crosstable(diabetes, gender, by = dm) %>% as_flextable()
crosstable(diabetes, frame, by = dm) %>% as_flextable()
crosstable(diabetes, insurance, by = dm) %>% as_flextable()
crosstable(diabetes, fh, by = dm) %>% as_flextable()
crosstable(diabetes, age_group, by = dm) %>% as_flextable() 
crosstable(diabetes, bmi_cat, by = dm) %>% as_flextable() 
crosstable(diabetes, chol_cat, by = dm) %>% as_flextable()

#trends seen in frame, insurance, bmi_cat, chol_cat
#sig dif in family hist

# continuous: chol, stab.glu, hdl, ratio, glyhb, age, height, weight, bp.1s, bp.1d, bp.2s, bp.2d, waist, hip, time.ppn
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



#linear relationships seen in:
# chol, ratio, age, all bp, waist, hip


# variable selection
# according to literature:
# predictors are age, family history, blood pressure, bmi, cholesterol
str(diabetes)

continuous <-diabetes[, c("chol", "hdl", "ratio", "age", "height", "weight", "bp.1s", "bp.1d", "waist", "hip", "bmi")]
cor(continuous, method = "spearman", use = "complete.obs")
pairs(~chol + hdl + ratio + age + height + weight + bp.1s + bp.1d + waist + hip + bmi, data = diabetes)


# sys and diastolic bp are known to be correlated

#bmi, waist, hip, weight are correlated. BMI will be selected as the evidence for BMI is strong.

#chol and ratio moderately correlated, hdl and ratio strongly correlated.
#shall not include hdl as rs does not appear linear. cant include both chol and ratio due to correlation.

#methods to assess quality of model:predictive power and goodness of fit
#Measures of predictive power:
#R squared measures <- measures proportion of variance that can be explained by predictor variables e.g. McFadden(pseudo)R-squared, 
#Discrimination: measure of how well the model can distinguish those who have and dont have the outcome of interest. e.g. c statistic or area under area under the receiver operating characteristic (ROC)curve.
#ROC curve is a plot of sensitivity/ 1- specificity
#cstatistic = 0.5 indicates the model is only as good at predictive the outcome as ranndom chance. 1 indicates perfect prediction.

# Measures of goodness of fit:
# Deviance- a measure of the difference between the predicted and observed values.
# when using logistic regression, there are typically only 2 outcome variables. therefore instead of calculating deviance from the outcome variables, we use log odds which can take on any value and be mapped to probabilities.
# Residual deviance is given in the model summary. 
# To test whether each parameter in the model decreases the deviance by a significant amount for the degrees of freedom taken by the parameter, we can use a chi-square test.
# we can also use the Akaike Information Criterion (AIC) provided in the model summary. AIC is no use by itself but it can be used to compare to or more models, where a smaller value suggests a better goodness of fit.
# Another method to measure goodness of fit is the hosmer-lemeshow statistic and test.
# using this method, participants are grouped into typically 10 groups, according to their predicted values. 
# the observed number of outcomes are compared with the predicted number of outcomes using pearson's chi-square test. a large p value indicates that the model is a good fit.
# However, this method has limitations when the sample sizes are too small or too large. Another issue is that there is no good way of deciding on the number of groups to split the participants into.

# to verify the robustness of a model, it is sometimes advisable to split the data into a training and testing set to verify that the model generates reproducible results in both datasets.
# However, this is not possible for this dataset as the sample size is small.


#fitting multiple logistic regression model.
#bp1.s or bp1.d
#bmi or weight or hip or waist
#chol or ratio

null_model<- glm(data = diabetes, dm~1, family = binomial(link = "logit"))

model <- glm(data= diabetes, dm~ age + gender + height +location + smoking + fh + bp.1s + bmi + chol + insurance, family = binomial(link = "logit"))
sum_model <- summary(model)
sum_model
exp(sum_model$coefficients)
exp(confint(model))

anova(model, test = "Chisq")


library(ResourceSelection)
HL <- hoslem.test(x = model$y, y = fitted(model), g = 10)
HL

HL2 <- hoslem.test(x = model2$y, y = fitted(model2), g = 10)
HL2

HL3 <- hoslem.test(x = model3$y, y = fitted(model3), g = 10)
HL3

# plot of predicted value's prevalence against observed value's prevalence
plot(x = HL$observed[, "y1"]/ (HL$observed[,"y1"] + HL$observed[, "y0"]),
     y = HL$expected[, "yhat1"]/ (HL$expected[, "yhat1"] + HL$expected[, "yhat0"]))

plot(x = HL2$observed[, "y1"]/ (HL2$observed[,"y1"] + HL2$observed[, "y0"]),
     y = HL2$expected[, "yhat1"]/ (HL2$expected[, "yhat1"] + HL2$expected[, "yhat0"]))

plot(x = HL3$observed[, "y1"]/ (HL3$observed[,"y1"] + HL3$observed[, "y0"]),
     y = HL3$expected[, "yhat1"]/ (HL3$expected[, "yhat1"] + HL3$expected[, "yhat0"]))



#noticed that blood pressure is not statistically significant in the model
#investigating for colinearity
cor.test(diabetes$bp.1s, diabetes$age)#
cor.test(diabetes$bp.1s, diabetes$height)
cor.test(diabetes$bp.1s, diabetes$bmi)#
cor.test(diabetes$bp.1s, diabetes$chol)#

cor.test(diabetes$bp.1d, diabetes$age)
cor.test(diabetes$bp.1d, diabetes$height)
cor.test(diabetes$bp.1d, diabetes$bmi)#
cor.test(diabetes$bp.1d, diabetes$chol)#

library(car)
vif(model)




#notice that sys blood pressure is significantly correlated with age, bmi and chol
#vif is still ok, but correlation will be reported in results.
#diastolic bp is significantly correlated with bmi and chol.
#non-significant variables will be eliminated from the model

model2 <- glm(data= diabetes, dm~ age + bp.1s + fh + bmi + chol, family = binomial(link = "logit"))
sum_model2 <- summary(model2)
sum_model2
exp(sum_model2$coefficients)
exp(confint(model2))

vif(model2)
#replace bp.1s with bp.1d
model3 <- glm(data = diabetes, dm~age + bp.1d + fh + bmi + chol, family = binomial(link = "logit"))
sum_model3 <- summary(model3)
sum_model3
exp(sum_model$coefficients)
exp(confint(model))

vif(model3)

#Calculating McFadden's Pseudo R^2
R2 <-1 - logLik(model)/logLik(null_model)
R2

R2_2 <- 1- loglik(model2)/loglik(null_model)
R2_2

R2_3 <- 1 - loglik(model3)/loglik(null_model)
R2_3


#calculating ROC and cstat
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

#goodness of fit improved as seen by reduced AIC values
#however predictive power reduced as seen by cstat and R2

#calculating predicted probability of participant having dm


#plot ROC curve


#chisq tests
anova(model, test = "Chisq")
anova(model2, test = "Chisq")
anova(model3, test = "Chisq")


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
