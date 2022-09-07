# Medication Cart Quality Improvement Project

## Background

The organization of medication carts are suboptimal as location of medications in the trolley are not standardized and there is room to improve the ergonomics of medication dispensing using the trolley. The current design of the medication cart dividers is such that columns of the dividers are screwed onto the base of the drawer, and horizontal dividers are fitted with pieces of cardboard or plastic. This design impedes efforts to upkeep the cleanliness of the medication cart as the dividers cannot be easily removed and reassembled for cleaning. Furthermore, improvement in the construction of the cart dividers can improve the clarity and presentation of items in the cart. In addition, the arrangement of items in the medication carts are not standardized across the different teams’ designated carts, which could reduce nurses’ familiarity with their placement and impair productivity. 
**The aim of this QI project is to improve the cleanliness, construction quality, ergonomics, readability, and safety of medication carts within 6 months.**

## Measures
Users were given a pre-test survey of 10 questions on a 5-point likert scale ranging from 1(strongly disagree) to 5(strongly agree) to assess their impression of the cart’s cleanliness, ergonomics, construction quality, readability of labels, and safety. 6 months after initiation of interventions, users were given a post-test survey with the same set of questions as the pre-test to evaluate their impression of the cart. Both surveys were administered via surveyplanet. 
Sample sizes of n = 32 and n = 29 were gathered from the pre-test and post-test surveys respectively. The data was exported from surveyplanet into an excel.csv file and subsequently analysed using R (version 4.2.1). 
Results from the questionnaires were grouped into their respective categories, i.e. cleanliness, ergonomics, construction quality, readability and safety. Their scores for each category were then summed and divided by the sample size to obtain the mean.

The survey questions include the following:
1. I frequently need to bend low or squat in order to reach the item I wish to get.
2. I frequently mistakenly open the wrong compartment, thinking I would find the intended item there.
3. The main drawer of our medication carts is easy to clean.					
4. Our supply of medications in the carts are stored in a hygienic environment.					
5. The medication cart is so disorganised that it increases risk of medication administration errors .					
6. Our medication carts are sufficiently organized to avoid errors during medication administration.					
7. The construction of compartment dividers in the main drawer is of good quality.					
8. The compartment dividers in the main drawer is flimsy					
9. The labelling of medications in the drawers are easy to read					
10. Similar looking medications are easy to tell apart as they are adequately separated					

*As the data collected is confidential, results will not be reported here*

First, the post-test dataset is cleaned and prepared for analysis
```
#Opening the relevant libraries
library(tidyverse)
library(RColorBrewer)

#Reading post-test dataset
kaizen_data_post<- read_csv("C:\\Users\\isaac\\OneDrive\\Documents\\Projects\\datasets\\medication_cart_kaizen_post.csv")
kaizen_data_post <- kaizen_data_post %>%
  
#selecting relevant data
  select(c(11:20)) %>%
```
columns from the excel sheet are renamed according to their respective question categories
```
  rename(
    ergonomics1 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (I frequently need to bend low or squat in order to reach the item I wish to get.)",
    ergonomics2 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (I frequently mistakenly open the wrong compartment, thinking I would find the intended item there.)",
    cleanliness1 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (The main drawer of our medication carts is easy to clean.)",
    cleanliness2 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (Our supply of medications in the carts are stored in a hygienic environment.)",
    safety1 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (The medication cart is so disorganised that it increases risk of medication administration errors	.)",
    safety2 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (Our medication carts are sufficiently organized to avoid errors during medication administration.)",
    construction_quality1 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (The construction of compartment dividers in the main drawer is of good quality.)",
    construction_quality2 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (The compartment dividers in the main drawer is flimsy)",
    readability1 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (The labelling of medications in the drawers are easy to read)",
    readability2 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (Similar looking medications are easy to tell apart as they are adequately separated)"
      ) %>%
```
As some survey questions were phrased in different wording, the scores for these questions were reverse coded.
```
#reverse coding
  mutate(
      ergonomics2 = abs(5-ergonomics2),
      safety1 = abs(5 - safety1),
      construction_quality2 = abs(5 - construction_quality2)
        )%>%
```
There are two questions to assess each category. The scores from both questions are added to aggregate the total score for each category.
```
#calculating overall scores by category
  mutate(
      Ergonomics2 = ergonomics1 + ergonomics2,
      Cleanliness2 = cleanliness1 + cleanliness2,
      Safety2 = safety1 + safety2,
      Construction_Quality2 = construction_quality1 + construction_quality2,
      Readability2 = readability1 + readability2
        ) %>%
```
one NA value was found in the readability category. Admittedly, this was an error in survey creation. participants should not have been allowed to complete the survey with an unanswered question. The missing value is replaced by the average of the remaining readability scores.

```
#replacing NA value in readability
  mutate(Readability2 = replace_na(Readability2, mean(Readability2, na.rm = TRUE))) %>%
```
The data is transformed to prepare for visualization
```
#selecting by categories
  select(Ergonomics2:Readability2) %>%
  
#Transforming to plotable format
  pivot_longer(cols = 1:5, names_to = "variables", values_to = "total_score")
  ```
 The same treatment is repeated on the pre-test dataset
 ```
 #Reading pre-test dataset
kaizen_data_pre <- read_csv("C:\\Users\\isaac\\OneDrive\\Documents\\Projects\\datasets\\medication_cart_kaizen_pre.csv")

kaizen_data_pre <- kaizen_data_pre %>%
  
#selecting relevant data
  select(c(11:20)) %>%
  
#renaming the columns
  rename(
    ergonomics1 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (I frequently need to bend low or squat in order to reach the item I wish to get.)",
    ergonomics2 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (I frequently mistakenly open the wrong compartment, thinking I would find the intended item there.)",
    cleanliness1 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (The main drawer of our medication carts is easy to clean.)",
    cleanliness2 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (Our supply of medications in the carts are stored in a hygienic environment.)",
    safety1 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (The medication cart is so disorganised that it increases risk of medication administration errors	.)",
    safety2 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (Our medication carts are sufficiently organized to avoid errors during medication administration.)",
    construction_quality1 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (The construction of compartment dividers in the main drawer is of good quality.)",
    construction_quality2 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (The compartment dividers in the main drawer is flimsy)",
    readability1 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (The labelling of medications in the drawers are easy to read)",
    readability2 = "Q1 - From a scale of 1(strongly disagree) to 5 (strongly agree), tell us how strongly you agree/disagree with the following statements regarding our medication carts: (Similar looking medications are easy to tell apart as they are adequately separated)"
         ) %>%
  
#reverse coding
  mutate(
    ergonomics2 = abs(5-ergonomics2), 
    safety1 = abs(5 - safety1),
    construction_quality2 = abs(5 - construction_quality2)
      ) %>%
  
#calculating overall scores by category
  mutate(
    Ergonomics = ergonomics1 + ergonomics2, 
    Cleanliness = cleanliness1 + cleanliness2,
    Safety = safety1 + safety2,
    Construction_Quality = construction_quality1 + construction_quality2,
    Readability = readability1 + readability2
         ) %>%
  
#replacing NA value in readability
  mutate(Readability = replace_na(Readability, mean(Readability, na.rm = TRUE))) %>%
  
#selecting by categories
  select(Ergonomics:Readability) %>%

#Transforming to plotable format
  pivot_longer(cols = 1:5, names_to = "variables", values_to = "total_score")
  ```
 
