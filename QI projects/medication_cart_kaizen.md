# Medication Cart Quality Improvement Project

## Background

The organization of medication carts are suboptimal as location of medications in the trolley are not standardized and there is room to improve the ergonomics of medication dispensing using the trolley. The current design of the medication cart dividers is such that columns of the dividers are screwed onto the base of the drawer, and horizontal dividers are fitted with pieces of cardboard or plastic. This design impedes efforts to upkeep the cleanliness of the medication cart as the dividers cannot be easily removed and reassembled for cleaning. Furthermore, improvement in the construction of the cart dividers can improve the clarity and presentation of items in the cart. In addition, the arrangement of items in the medication carts are not standardized across the different teams’ designated carts, which could reduce nurses’ familiarity with their placement and impair productivity. </br>
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
      Ergonomics = ergonomics1 + ergonomics2,
      Cleanliness = cleanliness1 + cleanliness2,
      Safety = safety1 + safety2,
      Construction_Quality = construction_quality1 + construction_quality2,
      Readability = readability1 + readability2
        ) %>%
```

one NA value was found in the readability category. Admittedly, this was an error in survey creation. participants should not have been allowed to complete the survey with an unanswered question. The missing value is replaced by the average of the remaining readability scores.

```
#replacing NA value in readability
  mutate(Readability = replace_na(Readability, mean(Readability, na.rm = TRUE))) %>%
```

The data is transformed to prepare for analysis. we arrange the data into two columns: variables and total_score. This is so that we may analyse the data later. To plot the data, we first sum the total scores in each variable then divide by the sample size. This gives us the average scores for each category.
```
#selecting by categories
  select(Ergonomic:Readability) %>%
  pivot_longer(cols = 1:5, names_to = "variables", values_to = "total_score")
  
 #obtaining average scores by category for plotting
  kaizen_data_post_plot <- kaizen_data_post %>%
  group_by(variables) %>%
  summarize(total_score = sum(total_score)/ 29)
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
  pivot_longer(cols = 1:5, names_to = "variables", values_to = "total_score")

#obtaining average scores by category for plotting
kaizen_data_pre_plot <- kaizen_data_pre %>%
  group_by(variables) %>%
  summarize(total_score = sum(total_score)/ 32) 
  ```
  
Finally,the data is primed for plotting. geom_point is used for post-test data while geom_col is used for pre-test so that the difference in results are clearly visible. Furthermore, I've chosen an alpha of 0.5 to add some transparency so that points can be seen even if they overlap with the columns. The colors and theme were chosen as such to enhance clarity and visual appeal. The labels on the x axis are dodged to avoid clutter.
```
#plotting data
ggplot() + geom_point(data = kaizen_data_post_plot, aes(variables, total_score, col = variables)) +
  geom_col(data = kaizen_data_pre_plot, aes(variables, total_score, fill = variables), col = "black", alpha = 0.5) +
  labs(title = "Medication cart kaizen", x = "Assessment Categories", y = "Average Scores", color = "Post-test", fill = "Pre-test") +
  scale_fill_brewer(palette = "Dark2") +
  scale_color_brewer(palette = "Dark2") +
  guides(x = guide_axis(n.dodge = 2)) +
  theme_classic()
  ```
  
Now we'd like to statistically compare the pre-test and post-test results. Before getting to that, variables in the posttest dataset were renamed so that those in pre and post tests can be differentiated. Then, both datasets were combined and filtered for each category so that they may be analysed individually.

```
#Differentiating pre-test and post-test data
kaizen_data_post$variables[kaizen_data_post$variables == "Ergonomics"] <- "Ergonomics2"
kaizen_data_post$variables[kaizen_data_post$variables == "Cleanliness"] <- "Cleanliness2"
kaizen_data_post$variables[kaizen_data_post$variables == "Construction_Quality"] <- "Construction_Quality2"
kaizen_data_post$variables[kaizen_data_post$variables == "Readability"] <- "Readability2"
kaizen_data_post$variables[kaizen_data_post$variables == "Safety"] <- "Safety2"

#combining pre-test and post-test datasets
kaizen_data_prepost <- rbind(kaizen_data_pre, kaizen_data_post)

#filter by categories
ergonomics_result <- kaizen_data_prepost %>% filter(variables == "Ergonomics" | variables == "Ergonomics2")
cleanliness_result <- kaizen_data_prepost %>% filter(variables == "Cleanliness" | variables == "Cleanliness2")
construction_result <- kaizen_data_prepost %>% filter(variables == "Construction_Quality" | variables == "Construction_Quality2")
readability_result <- kaizen_data_prepost %>% filter(variables == "Readability" | variables == "Readability2")
safety_result <- kaizen_data_prepost %>% filter(variables == "Safety" | variables == "Safety2")
 ```
 
Now the dataset is ready for statistical analysis. First, we'll take a quick look at the distribution of results in each category, then perform Mann Whitney U tests to compare each pairs of pre and post-test data. Ideally, wilcoxon signed ranked test or paired samples t tests should be used for pre and posttest data. Unfortunately, the pre and posttest samples are unmatched due to unforeseen movement of staff. Here, Mann Whitney U tests are used instead of independent sample t tests as the sample sizes are small (n= 32) and (n= 29)
```
#Summary of results
table(ergonomics_result)
summary(ergonomics_result)
hist(ergonomics_result$total_score, main = "Ergonomics", ylab = "Total score")

table(cleanliness_result)
summary(cleanliness_result)
hist(cleanliness_result$total_score, main = "Cleanliness", ylab = "Total score")

table(construction_result)
summary(construction_result)
hist(construction_result$total_score, main = "Construction Quality", ylab = "Total score")

table(readability_result)
summary(readability_result)
hist(readability_result$total_score, main = "Readability", ylab = "Total score")

table(safety_result)
summary(safety_result)
hist(safety_result$total_score, main = "Safety", ylab = "Total score")

#Mann whitney U tests for ordinal data
ergonomics_wil <- wilcox.test(data = ergonomics_result, paired = FALSE, exact = FALSE, total_score ~ variables)
ergonomics_wil
cleanliness_t <- wilcox.test(data = cleanliness_result, paired = FALSE, exact = FALSE, total_score ~ variables)
cleanliness_t
construction_wil <- wilcox.test(data = construction_result, paired = FALSE, exact = FALSE, total_score ~ variables)
construction_wil
readability_wil <- wilcox.test(data = readability_result, paired = FALSE, exact = FALSE, total_score ~ variables)
readability_wil
safety_wil <- wilcox.test(data = safety_result, paired = FALSE, exact = FALSE, total_score ~ variables)
safety_wil
```
