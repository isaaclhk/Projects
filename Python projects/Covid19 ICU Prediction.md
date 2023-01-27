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

Based on the output from 'info', we see that majority of the features are floats, and there is a significant proportion of null values. To better understand the distribution of data in this dataset, we start by examining the non-float features individually.

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

![icu_bar](https://user-images.githubusercontent.com/71438259/215051195-6ce24646-b82e-4c89-b5a9-4aec7c144f9e.png)

![icu_line](https://user-images.githubusercontent.com/71438259/215051280-f59b50e9-c4c9-430d-8769-c90fe1522352.png)

From the charts above, we observe that 195 out of 384 patients who were included in this study were eventually admitted into ICU. The rate of ICU admission is relatively consistent from 0 to 12 hours, and almost a third of the patients were admitted into ICU after more than 12 hours.

Based on the output of data.describe, we can see that majority of the features have been scaled by min max normalization. The only features that havent been scaled are 'AGE_PERCENTIL' and 'WINDOW'. Although tree based models like xgboost are invariant to monotonic transformation of features, the 'AGE_PERCENTIL' feature will be min max scaled to match the other features in the dataset. However, we will not scale the window feature as it will not be included as a potential predictor of ICU admission. The reason for its exclusion is that admission of patients to ICU is contingent upon the patients' medical condition, of which window is not an element.

```
#preprocessing and scaling
data['AGE_PERCENTIL'] = pd.factorize(data['AGE_PERCENTIL'])[0]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data['AGE_PERCENTIL'] = scaler.fit_transform(data[['AGE_PERCENTIL']])
data['AGE_PERCENTIL'].unique()
```

As the project aims to predict whether or not a patient will eventually require ICU admission based on his or her clinical parameters prior to admission, we will remove examples where the ICU status is positive. This will ensure that the machine learning model is trained on pre-ICU admission data. After which, the 'ICU' column will be updated to indicate whether or not the patient in each example is eventually admitted to ICU.

```
#update ICU column. If patient is eventually admitted into ICU, ICU = 1
dataf.rename(columns = {'ICU': 'ICU_old'}, inplace = True)
dataf.columns
df = pd.merge(dataf, ICU, how = 'left', on = 'PATIENT_VISIT_IDENTIFIER')
df.shape
df['ICU'].value_counts() #945 is correct because 189 patients didnt admit into ICU. 189 x 5 windows = 945
df.info(verbose = True, show_counts = True)
```

we note from the info ouput that there is a significant portion of null values in the labs and vital signs monitoring features. This is likely because labs and vital signs of these patients were not monitored at strict 2 hourly intervals. The missing data will first be imputed by forward filling. Forward filling is the appropriate method of imputation as it mimics the actual clinical scenario where the latest results are interpreted. The remaining null values will then be imputed by backward filling. Furthermore, examples will be grouped by 'PATIENT_VISIT_IDENTIFIER' to prevent the clinical parameters of one patient from spilling over to adjacent patients during imputation.

```
#impute missing data using forward and backward fill by patient
df = df.groupby('PATIENT_VISIT_IDENTIFIER').apply(lambda x: x.ffill().bfill())
df.info(verbose = True, show_counts = True)
```

Finally, features that are impertinent to the model are dropped, and the outcome variable separated.
```
#drop irrelevant columns
df.drop(columns = ['ICU_old', 'WINDOW', 'PATIENT_VISIT_IDENTIFIER'], inplace = True)
list(df.columns)

#separating target and features
X = df.drop('ICU', axis = 1)
Y = df['ICU']
```
## The model
xgboost is the chosen model to generate predictions for this project. The model will be cross validated to obtain a robust estimate of model prediction performance. In addition, the hyperparameters will be tuned with cross validated randomized grid search. Hence, each split of the outerloop is trained with optimal parameters identified from cross validating the inner loop. Finally, feature importance will be evaluated using shapley additive exaplanations (SHAP).
</br></br>
We begin by importing the relevant libraries.

```
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import shap
```

Next, we establish the CV scheme for the outer and inner loops and specify the hyperparameters to be tuned.
```
#Establish CV scheme
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

## Establish inner CV for parameter optimization
cv_inner = StratifiedKFold(n_splits=3, shuffle= True, random_state = 42)

#hyperparameters to be tuned
params = {}
params['max_depth'] = range(3, 10)
params['subsample'] = np.arange(0.5, 1, 0.05)
params['colsample_bytree'] = np.arange(0.5, 1, 0.05)
params['n_estimators'] = np.arange(50, 300, 50)
params['lambda'] = np.arange(0, 1, 0.05)
params['gamma'] = np.arange(0, 0.2, 0.05)
params['eta'] = [0.01, 0.05, 0.1, 0.2]
```



```
------ Fold Number: 1
optimized hyperparameters for Fold Number 0 are: {'subsample': 0.9000000000000004, 'n_estimators': 250, 'max_depth': 6, 'lambda': 0.4, 'gamma': 0.0, 'eta': 0.2, 'colsample_bytree': 0.8500000000000003}
Prediction accuracy of Fold Number 0 is: 
            training accuracy: 99.56
            testing accuracy: 97.15
ntree_limit is deprecated, use `iteration_range` or model slicing instead.

------ Fold Number: 2
optimized hyperparameters for Fold Number 1 are: {'subsample': 0.6500000000000001, 'n_estimators': 250, 'max_depth': 5, 'lambda': 0.2, 'gamma': 0.1, 'eta': 0.1, 'colsample_bytree': 0.7500000000000002}
Prediction accuracy of Fold Number 1 is: 
            training accuracy: 99.91
            testing accuracy: 97.51

------ Fold Number: 3
ntree_limit is deprecated, use `iteration_range` or model slicing instead.
optimized hyperparameters for Fold Number 2 are: {'subsample': 0.6000000000000001, 'n_estimators': 250, 'max_depth': 5, 'lambda': 0.6000000000000001, 'gamma': 0.1, 'eta': 0.1, 'colsample_bytree': 0.7000000000000002}
Prediction accuracy of Fold Number 2 is: 
            training accuracy: 99.91
            testing accuracy: 98.22

------ Fold Number: 4
ntree_limit is deprecated, use `iteration_range` or model slicing instead.
optimized hyperparameters for Fold Number 3 are: {'subsample': 0.6000000000000001, 'n_estimators': 150, 'max_depth': 6, 'lambda': 0.15000000000000002, 'gamma': 0.05, 'eta': 0.1, 'colsample_bytree': 0.7000000000000002}
Prediction accuracy of Fold Number 3 is: 
            training accuracy: 99.91
            testing accuracy: 98.22

------ Fold Number: 5
ntree_limit is deprecated, use `iteration_range` or model slicing instead.
optimized hyperparameters for Fold Number 4 are: {'subsample': 0.6500000000000001, 'n_estimators': 150, 'max_depth': 5, 'lambda': 0.8, 'gamma': 0.05, 'eta': 0.2, 'colsample_bytree': 0.9500000000000004}
Prediction accuracy of Fold Number 4 is: 
            training accuracy: 99.91
            testing accuracy: 99.29
ntree_limit is deprecated, use `iteration_range` or model slicing instead.
```