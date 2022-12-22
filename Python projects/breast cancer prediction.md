# Breast Cancer Prediction
## Background
The aim of this project is to build a logistic regression model that will accurately predict whether or not a breast tumor is benign or malignant based on cell nuclei characteristics. While there are powerful libraries available to help us perform logistic regression efficiently, the model in this project will be built from scratch as an exercise to build intuition and understanding of how such libraries work under the hood. The results will then be verified against the model built using the sklearn library.  
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

### code

Importing relevant libraries and loading the dataset.

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
To ensure that there is no multicollinearity, correlations between the features are visualized using a correlation matrix.

```
#checking for multicollinearity
corr = df.corr().round(2)
#creating triangular heatmap
mask = np.triu(np.ones_like(corr, dtype = np.bool))
sns.heatmap(corr, annot = True, mask = mask, xticklabels = True, yticklabels = True)
```
