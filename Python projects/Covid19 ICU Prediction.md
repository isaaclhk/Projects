# Covid-19 ICU Prediction

## Background
Covid-19 is a virus that was first identified in Wuhan, China, in 2019. The disease quickly spread around the globe, resulting in a pandemic. The virus causes acute respiratory infections and can lead to complications such as respiratory distress syndrome and sepsis. During the height of the pandemic, ICU beds in numerous hospitals filled up quickly, and there was an urgent need to forecast bed demand to optimize resources. 
</br></br>
To assist clinicians in making faster and more accurate decisions, Hospital Sirio-Libanes in Sao Paulo, Brazil leveraged predictive analystics to enhance their clinical decision support and triaging processes. This enabled them to identify and give medical attention to patients that need ICU care most urgently. The Sírio-Libanês data for AI and Analytics by Data Intelligence Team has made the data used in their analysis available to the public, and this is the dataset that will be analysed in this project.
</br></br>
The aim of this project is to predict whether or not a patient is likely to eventually require ICU admission based on clinical data obtained from the patient while he or she is not in ICU.

### About the dataset
The dataset used in this project and its descriptions are available on [kaggle](https://www.kaggle.com/code/afamos/covid-19-icu-admission-prediction-92-accuracy/data).</br>
It contains anonymized data from Hospital Sírio-Libanês, São Paulo and Brasilia. All data were anonymized following the best international practices and recommendations.
Data has been cleaned and scaled by column according to Min Max Scaler to fit between -1 and 1.

**Available data**</br>
Patient demographic information (03) </br>
Patient previous grouped diseases (09)</br>
Blood results (36)</br>
Vital signs (06)</br>
In total there are 54 features, expanded when pertinent to the mean, median, max, min, diff and relative diff.</br></br>

diff = max - min</br>
relative diff = diff/median</br>

## Data cleaning and Exploratory Analysis
```
import pandas as pd
import numpy as np

data = pd.read_excel("C:/Users/isaac/OneDrive/Documents/Projects/datasets/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")

##getting to know the dataset
data.shape
data.info(verbose = True, show_counts = True)
```

Based on the output from 'info', we see that majority of the features are floats. To better understand the distribution of data in this dataset, we start by examining the non-float features individually.

```
data['AGE_ABOVE65'].value_counts(dropna = False)
data['AGE_PERCENTIL'].value_counts(dropna = False)
data['GENDER'].value_counts(dropna = False)
data['WINDOW'].value_counts(dropna = False)
data['ICU'].value_counts(dropna = False)
data['HTN'].value_counts(dropna = False)
data['IMMUNOCOMPROMISED'].value_counts(dropna = False)
data['OTHER'].value_counts(dropna = False)
```

By looking at the above features, we note that 'HTN', 'IMMUNOCOMPROMISED', and 'OTHER' consists of 5 NaN values. 
we investigate further by finding where these NaN values originate.

```
#investigate NaN values
data.loc[data['HTN'].isnull() == True]
data.loc[data['HTN'].isnull() == True]
data.loc[data['OTHER'].isnull() == True]
```
After investigation, we find that these 5 NaN values all come from the same patient: PATIENT_VISIT_IDENTIFIER 199. </br>
The data from this particular patient looks suspiciously empty as it provided us with no data apart from age and gender. As there is only one patient with almost no data, this patient will be dropped from the dataset.

```
#drop patient 199 as it contains no data
data.drop(index = data.index[995:1000], inplace = True)
data.loc[data['PATIENT_VISIT_IDENTIFIER'] == 199]
```

Next, we examine the remaining features' distributions.

```
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

data.describe()

#Distribution of paitents admitted to ICU
ICU = (data.groupby(by = 'PATIENT_VISIT_IDENTIFIER').sum()['ICU'] > 0).reset_index()*1
ICU.head()
ICU['ICU'].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style = 'darkgrid')
ax = sns.countplot(data = ICU, x = 'ICU')
ax.bar_label(container=ax.containers[0])
plt.title('ICU Admissions')
plt.xticks(ticks = [0,1], labels = ['not admitted', 'admitted'])
plt.close()

ICU_plot = data.groupby(by = 'WINDOW').sum()['ICU']
sns.lineplot(data = ICU_plot, marker = 'o')
plt.title('ICU Admission Window')
plt.close()
```






