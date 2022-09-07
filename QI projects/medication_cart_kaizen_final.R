#hi

#Opening the relevant libraries
library(tidyverse)
library(RColorBrewer)

#Reading post-test dataset
kaizen_data_post<- read_csv("C:\\Users\\isaac\\OneDrive\\Documents\\Projects\\datasets\\medication_cart_kaizen_post.csv")
kaizen_data_post <- kaizen_data_post %>%
  
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
        )%>%
  
#calculating overall scores by category
  mutate(
      Ergonomics2 = ergonomics1 + ergonomics2,
      Cleanliness2 = cleanliness1 + cleanliness2,
      Safety2 = safety1 + safety2,
      Construction_Quality2 = construction_quality1 + construction_quality2,
      Readability2 = readability1 + readability2
        ) %>%
  
#replacing NA value in readability
  mutate(Readability2 = replace_na(Readability2, mean(Readability2, na.rm = TRUE))) %>%
  
#selecting by categories
  select(Ergonomics2:Readability2) %>%
  
#Transforming to plotable format
  pivot_longer(cols = 1:5, names_to = "variables", values_to = "total_score")


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


#combining pre-test and post-test datasets
kaizen_data_prepost <- rbind(kaizen_data_pre, kaizen_data_post)

#filter by categories
ergonomics_result <- kaizen_data_prepost %>% filter(variables == "Ergonomics" | variables == "Ergonomics2")
cleanliness_result <- kaizen_data_prepost %>% filter(variables == "Cleanliness" | variables == "Cleanliness2")
construction_result <- kaizen_data_prepost %>% filter(variables == "Construction_Quality" | variables == "Construction_Quality2")
readability_result <- kaizen_data_prepost %>% filter(variables == "Readability" | variables == "Readability2")
safety_result <- kaizen_data_prepost %>% filter(variables == "Safety" | variables == "Safety2")

#Summary of results
table(ergonomics_result)
summary(ergonomics_result)

table(cleanliness_result)
summary(cleanliness_result)

table(construction_result)
summary(construction_result)

table(readability_result)
summary(readability_result)

table(safety_result)
summary(safety_result)

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


#Summarize for plotting
kaizen_data_post_plot <- kaizen_data_post %>% group_by(variables) %>%
  summarize(total_score = sum(total_score)/ 29) %>%
  mutate(variables = str_replace(variables, "2", ""))

kaizen_data_pre_plot <- kaizen_data_pre %>% group_by(variables) %>%
  summarize(total_score = sum(total_score)/ 32) 
kaizen_data_pre_plot

#plotting data
ggplot() + geom_point(data = kaizen_data_post_plot, aes(variables, total_score, col = variables)) +
  geom_col(data = kaizen_data_pre_plot, aes(variables, total_score, fill = variables), col = "black") +
  labs(title = "Medication cart kaizen", x = "Assessment Categories", y = "Average Scores", color = "Post-test", fill = "Pre-test") +
  scale_fill_brewer(palette = "Dark2") +
  scale_color_brewer(palette = "Dark2") +
  guides(x = guide_axis(n.dodge = 2)) +
  theme_classic()
  

