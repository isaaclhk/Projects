# Breast Cancer Prediction
## Background
The aim of this project is to build a logistic regression model that will accurately predict whether or not a breast tumor is benign or malignant based on cell nuclei characteristics. While there are powerful libraries available to help us perform logistic regression efficiently, the model in this project is built from scratch as an exercise to build intuition and understanding of how such libraries work under the hood. The results were then be verified against the model built using the sklearn library.  
</br>
### [About Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server:
ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

Attribute Information:

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

## Exploratory data analysis

We begin by importing relevant libraries and loading the dataset before describing and visualizing the data.

```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/isaac/OneDrive/Documents/Projects/datasets/data.csv')
print(df.head())

df.info()
```

id is dropped beacause it is not germane to the analysis, 'Unnamed: 32' is also dropped because it consists of only null values.

```
#check that the null column was dropped, note that all are floats except diagnosis
df.info()
df.shape
```

The dependent variable is examined to understand the distribution between malignant and benign tumors in the dataset.

```
#examine diagnosis variable
df['diagnosis'].unique()
df['diagnosis'].value_counts()
#M = malignant, B = benign, convert to 1 = malignant, 0 = benign
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

#visualize malignant vs benign
sns.countplot(x = 'diagnosis', data = df)
plt.title('Malignant vs Benign')
plt.show()
```

![Figure_1](https://user-images.githubusercontent.com/71438259/209077374-0e57a3b4-efb1-4bd8-9755-0d06ed16297b.png)

There are 357 benign tumors and 212 malignant tumors in the dataset. </br>
Next, the remaining features are described and visualized using scatterplot matrices to identify patterns and trends. </br>
3 sets of scatterplot matrices are made to compare means, standard errors and worsts respectively.

```
#describe the data
pd.options.display.max_columns= df.shape[1]
df.describe()

#visualize mean
df.columns
cols1 = df.loc[:, 'diagnosis': 'fractal_dimension_mean']
print(cols1.head())

sns.pairplot(data = cols1, hue = 'diagnosis')

#visualize standard error
cols2 = df.loc[:, 'radius_se': 'fractal_dimension_se']
cols2['diagnosis'] = df['diagnosis']
print(cols2.head())

sns.pairplot(data = cols2, hue = 'diagnosis')

#visualize worst
cols3 = df.loc[:, 'radius_worst': 'fractal_dimension_worst']
cols3['diagnosis'] = df['diagnosis']
print(cols3.head())

sns.pairplot(data = cols3, hue = 'diagnosis')
```
**Mean
![mean](https://user-images.githubusercontent.com/71438259/209077971-a90cf850-2154-4547-ab4b-373ad0cd24ab.png)
</br>
Standard Error
![se](https://user-images.githubusercontent.com/71438259/209077986-83ab0903-2186-4417-9e04-9149758900a4.png)
</br>
Worst**
![worst](https://user-images.githubusercontent.com/71438259/209077993-904b2827-1459-446a-be58-057c0101bc2c.png)
</br>
From the above plots, we note that the relationships between radius, perimeter and area attributes appear to be strongly linear. Concavity, concave_points and compactness may also possibly be multicollinear. </br>
To ensure that there is no multicollinearity, correlations between the features are visualized using a correlation matrix.

```
#checking for multicollinearity
corr = df.corr().round(2)
#creating triangular heatmap
mask = np.triu(np.ones_like(corr, dtype = np.bool))
sns.heatmap(corr, annot = True, mask = mask, xticklabels = True, yticklabels = True)
```

![corrmat](https://user-images.githubusercontent.com/71438259/209079477-c24ca2f7-090e-49f4-a0d8-0f99db60d6c8.png)

Based on the correlation values, the worsts and means are highly correlated. This is expected as the worsts are a subset of means. To avoid multicollinearity, attributes of the worsts columns will be dropped. In addition, radius, perimeter and area attributes are confirmed to be multicollinear. This is expected as these attributes are related measures of cell size. As the cell's area and perimeter can both be determined by its radius, the perimeter and radius attributes will be discarded from the analysis. Lastly, compactness, concavity and concave points is also highly correlated. This is also expected given that they are all indicators of cell shape. As compactness is a measure of overall shape that would account for both concavity and number of concave points, compactness will be retained while concavity  and concave points are dropped from the analysis.

```
#drop worst
df.drop(df.loc[:,'radius_worst': 'fractal_dimension_worst'], axis = 1, inplace = True)
df.columns

#drop radius and perimeter
df.drop(['area_mean', 'perimeter_mean', 'area_se', 'perimeter_se'], axis = 1, inplace = True)
df.columns

#drop concavity and concave points
df.drop(['concavity_mean', 'concave points_mean', 'concavity_se', 'concave points_se'], axis = 1, inplace = True)
df.columns
```

After the multicollinear features are removed, the correlation matrix is plotted once again.

```
#replot correlation matrix
corr = df.corr().round(2)
mask = np.triu(np.ones_like(corr, dtype = np.bool))
sns.heatmap(corr, annot = True, mask = mask, xticklabels = True, yticklabels = True)
```

![corrmat2](https://user-images.githubusercontent.com/71438259/209107952-592189d4-0ced-4d16-9aea-0101d52e68bc.png)

## Logistic Regression

We begin by splitting the dataset into two parts: </br>
1. A training set
2. A testing set
</br>
The model will be fitted on the training set and executed on the testing set. This practice helps us to evaluate whether our model is truly able to predict unseen data, and tells us if the model could be over or underfit. For this analysis, 70% of the data will be used for training before the model is tested on the remaining 30%. Random state is set to 42 to obtain reproducible results.

```
#Assign x and y, split data
x = df.drop('diagnosis', axis = 1)
y = df['diagnosis']

#Split dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 42)

#examine shapes
x_train.shape
x_test.shape
y_train.shape
y_test.shape
```

Before we begin to train the model, it is often good practice to scale the features so that every feature falls within a finite, standard range. This process ensures that the steps for gradient descent are updated at a similar rate for all features. There are various methods of feature scaling. For this analysis, we will use z-score normalization which transforms each feature to have a mean of 0 and standard deviation of 1. The formula for z-score normalization is shown below:
</br>

![zscore](https://user-images.githubusercontent.com/71438259/209116595-a0a06d05-f859-4046-a8b6-89f19fc5a414.jpg)

```
#z-score normalization
def norm(X):    
    # find the mean value of each column/feature
    mu     = np.mean(X, axis = 0)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis = 0)                  # sigma will have shape (n,)
    # element-wise, subtract mean for that column from each example, divide standard deviation
    X_norm = (X - mu) / sigma   

    return X_norm
    
#normalize independent variables
x_train_norm = norm(x_train)
x_test_norm = norm(x_test)
```

For logistic regression, the model is represented as
![logreg](https://user-images.githubusercontent.com/71438259/209117878-a85c1514-06d9-4302-9ce1-a4c0a49a31bc.jpg)


