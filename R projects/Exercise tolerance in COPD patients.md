# Predictors of Exercise Tolerance in Patients with COPD
## Background
This is a fictitious dataset obtained from the 2nd course of the Statistical Analysis with R for Public Health specialization: Linear Regression in R for public health.
The dataset can be found [here](https://github.com/isaaclhk/Projects/blob/main/datasets/COPD.csv).
The outcome variable for this analysis is MWT1Best, a measure of walking distance. 
we want to identify factors that predict walking ability in patients with COPD and produce a regression model that will best quantify the impact of these factors.

### Variables in the dataset
1. Age - Age in years
2. Packhistory - number of packs of cigarettes smoked per day * number of smoking years
3. COPD severity- levels = c(mild, moderate, severe, very severe)
4. MWT1 - Distance covered over first attempt of 6-minute walking test
5. MWT2- Distance covered over second attempt of 6- minute walking test
6. MWT1Best - Best performance between MWT1 and MWT2
7. FEV1- forced expiratory volume for the first second of exhalation
8. FEV1PRED - percentage of FEV1 relative to predicted value
9. FVC- forced vital capacity (total volume of air exhaled during FEV)
10. FVCPRED- percentage of FVC relative to predicted value
11. CAT- COPD Assessment Test, a questionnaire to quantify the impact of COPD on a person's life
12. HAD- Hospital Anxiety and Depression Scale, a questionnaire to measure anxiety and depression
13. SGRQ- St George's Respiratory Questionnaire, measures impact on daily life and perceived well-being in patients with obstructive airways disease.
14. AGEquartiles- Age variable categorised into quartiles
15. COPD- Indication of COPD severity, levels = 1:4 (mild to very severe)
16. Gender- levels = 0(female), 1(male)
17. Smoking- levels = 0(non-smokers), 1(ex-smokers), 2(current smokers) </br>
(18 to 22. ) Other comorbidities e.g. Diabetes, muscular, hypertension, AtrialFib, IHD- levels - 0(absent), 1(present)
