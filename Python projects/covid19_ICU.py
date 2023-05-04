import pandas as pd
import numpy as np

data = pd.read_excel("C:/Users/isaac/OneDrive/Documents/Projects/datasets/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")

##getting to know the dataset
data.shape
data.info(verbose = True, show_counts = True)

data['AGE_ABOVE65'].value_counts(dropna = False)
data['AGE_PERCENTIL'].value_counts(dropna = False)
data['GENDER'].value_counts(dropna = False)
data['WINDOW'].value_counts(dropna = False)
data['ICU'].value_counts(dropna = False)
data['HTN'].value_counts(dropna = False)
data['IMMUNOCOMPROMISED'].value_counts(dropna = False)
data['OTHER'].value_counts(dropna = False)

#investigate NaN values
data.loc[data['HTN'].isnull() == True]
data.loc[data['HTN'].isnull() == True]
data.loc[data['OTHER'].isnull() == True] #note that all NaN values are from the same patient: identifier 199

#drop patient 199 as it contains no data
data.drop(index = data.index[995:1000], inplace = True)
data.loc[data['PATIENT_VISIT_IDENTIFIER'] == 199]

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

#preprocessing and scaling
data['AGE_PERCENTIL'] = pd.factorize(data['AGE_PERCENTIL'])[0]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data['AGE_PERCENTIL'] = scaler.fit_transform(data[['AGE_PERCENTIL']])
data['AGE_PERCENTIL'].unique()

#remove examples where ICU=1
dataf = data[data['ICU'] == 0].copy()
dataf.shape

#update ICU column. If patient is eventually admitted into ICU, ICU = 1
dataf.rename(columns = {'ICU': 'ICU_old'}, inplace = True)
dataf.columns
df = pd.merge(dataf, ICU, how = 'left', on = 'PATIENT_VISIT_IDENTIFIER')
df.shape
df['ICU'].value_counts() #945 is correct because 189 patients didnt admit into ICU. 189 x 5 windows = 945
df.info(verbose = True, show_counts = True)

#impute missing data using forward and backward fill by patient
df = df.groupby('PATIENT_VISIT_IDENTIFIER').apply(lambda x: x.ffill().bfill())
df.info(verbose = True, show_counts = True)

#drop irrelevant columns
df.drop(columns = ['ICU_old', 'WINDOW', 'PATIENT_VISIT_IDENTIFIER'], inplace = True)
list(df.columns)

#separating target and features
X = df.drop('ICU', axis = 1)
Y = df['ICU']


## Cross validation with randomizedsearch
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import shap

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


#bringing back variable names    
x_test_set = pd.DataFrame(x_test_set, columns = X.columns)

#shap plots
shap.summary_plot(shap_values, x_test_set, plot_type = 'bar', max_display = 7)
shap.summary_plot(shap_values, x_test_set, max_display = 7)
shap.dependence_plot('PCR_MEDIAN', shap_values, x_test_set, interaction_index = 'AGE_ABOVE65')



