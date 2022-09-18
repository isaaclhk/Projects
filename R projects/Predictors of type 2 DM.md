# Predictors of Type 2 Diabetes Mellitus

## Background
This analysis performed on a dataset obtained from the 3rd course of the Statistical Analysis with R for Public Health specialization: Logistic Regression in R for public health. The dataset can be found [here](https://github.com/isaaclhk/Projects/blob/main/datasets/diabetes.csv). The outcome variable for this analysis is dm, a binary variable indicating whether or not a participant is diagnosed with type 2 diabetes mellitus. The aim of this project is to identify predictors of Type 2 DM and quantify their risks.

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

## Code
First, the relevant libraries and dataset are loaded. 
The column names are changed to all lowercase letters for simplicity, then we have a brief overview of the dataset.

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

6 patients were found to meet the criteria for DM diagnosis, but are not diagnosed with DM. Where possible, this discrepency should be clarified with the researcher.
</br>
After examining the data, we note that there are 403 observations, and that variables meant to be categorical are characters or integers. These will be changed to factors. It is sometimes useful to categorize patients into groups, for example by BMI or cholesterol levels, as thresholds are often used for clinical decision-making. Categorizing can also make it easier to track proportions of patients who meet certain thresholds. For this analysis, age, BMI and cholesterol will be categorised.
We note that there are some columns and variables that are either not relevant to the research question or are not appropriate as candidate predictors.
stab.glu and glyhb will be removed because they are used as diagnostic criteria for diabetes, hence not appropriate to be used as a predictor for our outcome DM.
time.ppn will also be removed as it is not a candidate predictor.

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



