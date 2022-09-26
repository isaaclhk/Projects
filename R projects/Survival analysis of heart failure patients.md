# Survival analysis of heart failure patients

## Background

## Assumptions of cox proportional hazards regression

1. non-informative censoring- likelihood of observation being censored is not related to likelihood of the event occuring
2. independent survival times- survival time of each observation is not related to that in any other observation.
3. Hazard ratios are proportional- the ratio of the hazards for any two individuals is constant over time.
4. ln(hazard) is a linear function of the predictor variables
5. values of predictor variables dont change over time


death (0/1)

los (hospital length of stay in nights)

age (in years)

gender (1=male, 2=female)

cancer

cabg (previous heart bypass)

crt (cardiac resynchronisation device - a treatment for heart failure)

defib (defibrillator implanted)

dementia

diabetes (any type)

hypertension

ihd (ischaemic heart disease)

mental_health (any mental illness)

arrhythmias

copd (chronic obstructive lung disease)

obesity

pvd (peripheral vascular disease)

renal_disease

valvular_disease (disease of the heart valves)

metastatic_cancer

pacemaker

pneumonia

prior_appts_attended (number of outpatient appointments attended in the previous year)

prior_dnas (number of outpatient appointments missed in the previous year)

pci (percutaneous coronary intervention)

stroke (history of stroke)

senile

quintile (socio-economic status for patient's neighbourhood, from 1 (most affluent) to 5 (poorest))

ethnicgroup (see below for categories)

fu_time (follow-up time, i.e. time in days since admission to hospital) 

Ethnic group has the following categories in this extract:

1=white 

2=black 

3=Indian subcontinent 

8=not known 

9=other
