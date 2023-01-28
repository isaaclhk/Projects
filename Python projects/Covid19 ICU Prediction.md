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
## The Model
[XGboost](https://xgboost.readthedocs.io/en/stable/), a decision-tree based ensemble machine learning algorithm that uses a gradient boosting framework is the chosen model to generate predictions for this project. The model will be cross validated to obtain a robust estimate of model prediction performance. In addition, the hyperparameters will be tuned with cross validated randomized grid search. Hence, each split of the outerloop is trained with optimal parameters identified from cross validating the inner loop. Finally, feature importance will be evaluated using [SHapley additive exaPlanations (SHAP)](https://shap.readthedocs.io/en/latest/).
</br></br>
We begin by importing the relevant libraries.

```
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import shap
```

Next, we establish the CV scheme for the outer and inner loops and specify the hyperparameters to be tuned. For this analysis we will have 5 folds in the outer loop and 3 folds in the inner loop.
```
#Establish CV scheme
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

## Establish inner CV for parameter optimization
cv_inner = StratifiedKFold(n_splits=3, shuffle= True, random_state = 42)

#hyperparameters to be tuned
params = {}
params['max_depth'] = range(3, 10)
params['subsample'] = np.round(np.arange(0.5, 1, 0.05), 2)
params['colsample_bytree'] = np.round(np.arange(0.5, 1, 0.05), 2)
params['n_estimators'] = range(100, 401, 50)
params['lambda'] = np.round(np.arange(0, 1, 0.05), 2)
params['gamma'] = np.round(np.arange(0, 0.2, 0.05), 2)
params['eta'] = np.round(np.arange(0.1, 0.3, 0.05), 2)
```
Finally, we begin to train the model. As the outer loop is split into 5 folds for cross validation, the test sets from every fold are concatenated to form a complete dataset of every example that is included in the analysis. The shap values obtained from each fold are likewise concatenated. This allows us to calculate the total prediction accuracy and visualize shap values of the cross validated model from every fold combined.

```
#lists to append during loops
ix_training, ix_test = [], []
list_shap_values = []
list_x_test_sets = []
list_y_test_sets = []
prediction_list = []


# Loop through each fold and append the training & test indices to the empty lists above
for fold in CV.split(X, Y):
    ix_training.append(fold[0]), ix_test.append(fold[1])
    

for i, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test)):
    #Verbose
    print('\n------ Fold Number:',i + 1)
    x_train, x_test = X.iloc[train_outer_ix, :], X.iloc[test_outer_ix, :]
    y_train, y_test = Y.iloc[train_outer_ix], Y.iloc[test_outer_ix]
    
    list_x_test_sets.append(x_test)
    list_y_test_sets.append(y_test)
    
    estimator = XGBClassifier(objective = 'binary:logistic')
    search = RandomizedSearchCV(estimator, params, cv = cv_inner, scoring = 'accuracy', n_iter = 100)
    search.fit(x_train, y_train)
    print('Optimized hyperparameters:\n {}\n'.format(search.best_params_))

    model = XGBClassifier(objective = 'binary:logistic',    
                           early_stopping_rounds = 20,
                           eval_metric = 'error',
                           **search.best_params_)
    model.fit(x_train, y_train,
              eval_set = [(x_test, y_test)],    
              verbose = False)
    train_prediction = model.predict(x_train)
    prediction = model.predict(x_test)
    prediction_list.append(prediction)
    
    #print accuracy score for each fold
    train_accuracy = accuracy_score(y_train, train_prediction)
    accuracy = accuracy_score(y_test, prediction)
    print('training accuracy: {train_accuracy:.2f}\ntesting accuracy: {accuracy:.2f}'.format(
          train_accuracy = train_accuracy*100, accuracy = accuracy*100))
 
    # Use SHAP to explain predictions using best estimator 
    explainer = shap.TreeExplainer(model) 
    shap_values = explainer.shap_values(x_test)
    list_shap_values.append(shap_values)
    
    
#concatenating test sets and shapley values
x_test_set = list_x_test_sets[0]
y_test_set = list_y_test_sets[0]
shap_values = np.array(list_shap_values[0])
total_prediction = prediction_list[0]


for i in range(1,len(list_x_test_sets)):
    x_test_set = np.concatenate((x_test_set,list_x_test_sets[i]),axis=0)
    y_test_set = np.concatenate((y_test_set,list_y_test_sets[i]),axis=0)
    shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=0)
    total_prediction = np.concatenate((total_prediction, prediction_list[i]), axis=0)

```
 
### Model Output

```
------ Fold Number: 1
Optimized hyperparameters:
 {'subsample': 0.6, 'n_estimators': 350, 'max_depth': 5, 'lambda': 0.15, 'gamma': 0.0, 'eta': 0.2, 'colsample_bytree': 0.85}

training accuracy: 99.38
testing accuracy: 97.51
ntree_limit is deprecated, use `iteration_range` or model slicing instead.

------ Fold Number: 2
Optimized hyperparameters:
 {'subsample': 0.9, 'n_estimators': 250, 'max_depth': 6, 'lambda': 0.2, 'gamma': 0.0, 'eta': 0.25, 'colsample_bytree': 0.5}

training accuracy: 100.00
testing accuracy: 97.15

------ Fold Number: 3
ntree_limit is deprecated, use `iteration_range` or model slicing instead.
Optimized hyperparameters:
 {'subsample': 0.6, 'n_estimators': 300, 'max_depth': 3, 'lambda': 0.05, 'gamma': 0.15, 'eta': 0.25, 'colsample_bytree': 0.8}

training accuracy: 100.00
testing accuracy: 98.58

------ Fold Number: 4
ntree_limit is deprecated, use `iteration_range` or model slicing instead.
Optimized hyperparameters:
 {'subsample': 0.65, 'n_estimators': 400, 'max_depth': 5, 'lambda': 0.7, 'gamma': 0.15, 'eta': 0.2, 'colsample_bytree': 0.6}

training accuracy: 99.82
testing accuracy: 98.22
ntree_limit is deprecated, use `iteration_range` or model slicing instead.

------ Fold Number: 5
Optimized hyperparameters:
 {'subsample': 0.8, 'n_estimators': 350, 'max_depth': 3, 'lambda': 0.9, 'gamma': 0.0, 'eta': 0.2, 'colsample_bytree': 0.65}

training accuracy: 99.82
testing accuracy: 98.22
ntree_limit is deprecated, use `iteration_range` or model slicing instead.
```

By comparing the training and testing accuracy, we can assess how much a model is over or underfitting the data. We can also determine whether the range of searched parameters should be adjusted by looking at the optimized parameters for each fold.

### Model Evaluation

Finally, we evaluate the model's test accuracy and plot a confusion matrix to visualize the model's sensitivity, specificity, precision and recall.
```
#total prediction accuracy after cross validation
total_accuracy = accuracy_score(y_test_set, total_prediction)
print('The model\'s total test accuracy is %.2f' %(total_accuracy*100))

#visualizing confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_set, total_prediction)
sns.heatmap(cm, annot = True, linewidths = 1, 
            xticklabels = ['not admitted', 'admitted'], 
            yticklabels = ['not admitted', 'admitted'],
            fmt = 'g',
            cmap = 'Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted ICU admission')
plt.ylabel('True ICU admission')
```

output:
```
The model's total test accuracy is 97.94
```

![cm](https://user-images.githubusercontent.com/71438259/215240808-0e0b5a56-3698-46c0-8218-935c11506ea7.png)

**SHAP(SHapley Additive exPlanations)** </br></br>
[SHAP](https://www.researchgate.net/profile/Scott-Lundberg/publication/317062430_A_Unified_Approach_to_Interpreting_Model_Predictions/links/5a18eb21a6fdcc50ade7ed19/A-Unified-Approach-to-Interpreting-Model-Predictions.pdf) is an algorithm based on game theory that helps us to interpret and explain complex machine learning models. Shapley values are calculated by quantifying for each example, the contribution of each feature to the model's prediction. </br></br>
With SHAP, we are able to create plots that allow us to interpret feature importances and understand the outputs of predictive models intuitively.

```
#bringing back variable names    
x_test_set = pd.DataFrame(x_test_set, columns = X.columns)

#shap plots
shap.summary_plot(shap_values, x_test_set, plot_type = 'bar', max_display = 7)
shap.summary_plot(shap_values, x_test_set, max_display = 7)
shap.dependence_plot('PCR_MEDIAN', shap_values, x_test_set, interaction_index = 'AGE_ABOVE65')
```
![bar_plot](https://user-images.githubusercontent.com/71438259/215242222-911acb5e-4763-4964-a62f-c79b5b75d0fe.png)

The above output is a SHAP bar plot. This plot illustrates the mean absolute shapley values for each feature, which indicates the magnitute of the feature's impact on the model's prediction. In this plot, the top 7 features are sorted and displayed in descending order.

![summary_plot](https://user-images.githubusercontent.com/71438259/215242147-18c55637-6166-4ae3-b0a8-3506dd9df068.png)

Like the bar plot, the summary plot above outputs the top 7 features sorted in descending order. However, this plot provides greater detail. It allows us to see the distribution of shapley and feature values for each feature, and reflects the relationships between feature values and shapley values.

![dependence_plot](https://user-images.githubusercontent.com/71438259/215242406-e04ae3fa-f61b-4516-8ed8-f2a4a5987ef1.png)

In addition, dependence plots such as the one illustrated above allow us to focus on a particular feature and visualize interactions between features. In the plot above, we see the interaction between 'PCR_MEDIAN' and 'AGE_ABOVE65' and their relationships with 'PCR_MEDIAN' shap values.

## Conclusions
The results of this project have important clinical implications. By accurately forecasting whether or not a patient would require ICU admission hours in advance, hospitals can be better informed in making decisions for resource allocation. Furthermore, we have quantified the impact of each feature for predicting ICU admission using shap values. Clinicians can then identify the most important features to examine when evaluating the prognosis or status of each patient.

