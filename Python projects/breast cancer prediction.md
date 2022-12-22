# Breast Cancer Prediction
## Background
The aim of this project is to build a logistic regression model that will accurately predict whether or not a breast tumor is benign or malignant based on cell nuclei characteristics. While there are powerful libraries available to help us perform logistic regression efficiently, the model in this project is built from scratch as an exercise to build intuition and understanding of how such libraries work under the hood. The results were then verified against the model built using the sklearn library.  
</br>
Reference: The mathematical formulae written on this post were taken from [this online course](https://www.coursera.org/learn/machine-learning)

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

id is dropped because it is not germane to the analysis, 'Unnamed: 32' is also dropped because it consists of only null values.

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
Next, the remaining features are described and visualized using scatterplot matrices to identify patterns. </br>
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
</br> </br>
Standard Error
![se](https://user-images.githubusercontent.com/71438259/209077986-83ab0903-2186-4417-9e04-9149758900a4.png)
</br> </br>
Worst**
![worst](https://user-images.githubusercontent.com/71438259/209077993-904b2827-1459-446a-be58-057c0101bc2c.png)
</br></br>
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

Based on the correlation coefficients, the worsts and means are highly correlated. This is expected as the worsts are a subset of means. To avoid multicollinearity, attributes of the worsts columns will be dropped. In addition, radius, perimeter and area attributes are confirmed to be multicollinear. This is expected as these attributes are related measures of cell size. As the cell's area and perimeter can both be determined by its radius, the perimeter and radius attributes will be discarded from the analysis. Lastly, compactness, concavity and concave points are also highly correlated. This is also expected given that they are all indicators of cell shape. As compactness is a measure of overall shape that would account for both concavity and number of concave points, compactness will be retained while concavity  and concave points are dropped from the analysis.

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
The model will be fitted on the training set and executed on the testing set. This practice helps us to evaluate whether the model is truly able to predict unseen data, and tells us if the model could be over or underfit. For this analysis, 70% of the data will be used for training before the model is tested on the remaining 30%. Random state is set to 42 to obtain reproducible results.

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

![sigmoid](https://user-images.githubusercontent.com/71438259/209121022-399d130b-48a0-4009-be85-9ab73b72bd23.jpg)
</br>

```
#sigmoid function
def sigmoid(z):
    g = 1/(1 + np.exp(-z))
    return g
```

![costfunction](https://user-images.githubusercontent.com/71438259/209120704-13f7578a-0900-40ea-868a-b7c33d875b3a.jpg)
</br>
```
#cost funciton
def compute_cost(X, y, w, b):
    
    m, n = X.shape
    total_cost = 0
    
    for i in range(m):
        z_i = np.dot(X.iloc[i], w) + b
        f_wb_i = sigmoid(z_i)
        loss = (-y.iloc[i]*np.log(f_wb_i)) - (1 - y.iloc[i])*np.log(1 - f_wb_i)
        total_cost += loss
    total_cost = total_cost/m
        
    return total_cost
```

![gradient1](https://user-images.githubusercontent.com/71438259/209121558-4900355b-81e2-4021-b354-d0281262b801.jpg)
</br>
![gradient2](https://user-images.githubusercontent.com/71438259/209122843-7962eb70-81b1-4019-9987-33c16e696439.jpg)
</br>
```
#compute gradient
def compute_gradient(X, y, w, b):
    
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0
    
    for i in range(m):  
        z_wb = np.dot(X.iloc[i], w) + b
        f_wb = sigmoid(z_wb)

        dj_db_i = f_wb - y.iloc[i]
        dj_db += dj_db_i
        
        dj_dw_i = (f_wb - y.iloc[i])*X.iloc[i]
        dj_dw += dj_dw_i
    
    dj_db = dj_db/m
    dj_dw = dj_dw/m
    
    return dj_db, dj_dw

#gradient descent
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 

    # An array to store cost J
    J_history = []
    i_history = []

    for i in range(num_iters):
    
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in)

        # Update Parameters using w, b, alpha and gradient
        w_in -= alpha*dj_dw
        b_in -= alpha*dj_db
        
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in)
            J_history.append(cost)
            i_history.append(i)
            
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i%100 == 0 or i == (num_iters-1):
            print(f"Iteration {i}: Cost {float(J_history[i]):.6f}   ") 
        
            
    return w_in, b_in, J_history, i_history #J and i history for graphing
```

The gradient descent function that was created above outputs the weights and bias of the model after it has been fitted. </br>
Based on the weights and bias of the fitted model, we can compute the probability of malignancy for each example by applying g(w.x + b) as described above. </br>
</br>
![predict](https://user-images.githubusercontent.com/71438259/209127535-6abe1ed2-4e95-4018-81d9-76b95ed07b38.jpg)
</br>

```
#predict
def predict(X, w, b):
    
    m,n = X.shape
    p = np.zeros(m)
    
    for i in range(m):
        f_wb_i = sigmoid(np.dot(w, X.iloc[i]) + b)
        p[i] = f_wb_i >= 0.5
    
    return p
```

## Prediction results
Finally, we fit the model.

```
#fitting the model
initial_w = np.zeros(x_train_norm.shape[1])
initial_b = 0
iterations = 1000
alpha = 0.05


w, b, J_history, i_history = gradient_descent(x_train_norm, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print(w)
print(b)
```
Output:

```
w, b, J_history, i_history = gradient_descent(x_train_norm, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
Iteration 0: Cost 0.188628   
Iteration 100: Cost 0.175537   
Iteration 200: Cost 0.169519   
Iteration 300: Cost 0.166321   
Iteration 400: Cost 0.164379   
Iteration 500: Cost 0.163058   
Iteration 600: Cost 0.162078   
Iteration 700: Cost 0.161307   
Iteration 800: Cost 0.160674   
Iteration 900: Cost 0.160140   
Iteration 999: Cost 0.159685  

print(w)
radius_mean               2.406836
texture_mean              1.366080
smoothness_mean           0.793861
compactness_mean          1.277232
symmetry_mean             0.536747
fractal_dimension_mean   -0.303141
radius_se                 1.570828
texture_se               -0.452817
smoothness_se            -0.165410
compactness_se            0.005177
symmetry_se              -0.337414
fractal_dimension_se     -0.473001
Name: 149, dtype: float64

print(b)
-0.7267547221193337
```

The stepwise cost reduction process from gradient descent can be visualized in this plot

```
#plot cost against interations
sns.lineplot(x= i_history, y= J_history)
```
![gdplot](https://user-images.githubusercontent.com/71438259/209129011-bb3a5855-767a-4ffe-b142-dc46c8e9f55e.png)

Finally, we run the predict function and produce a confusion matrix to evaluate our model's accuracy.

```
#evaluating accuracy
print(f'Accuracy: {np.mean(p == y_test) * 100}')
from sklearn.metrics import confusion_matrix, accuracy_score
confmat = confusion_matrix(p, y_test)
confmat
```

output:

```
Accuracy: 95.90643274853801
array([[106,   5],
       [  2,  58]], dtype=int64)
       
#visualizing confusion matrix
sns.heatmap(confmat, annot = True, linewidths = 1, 
            xticklabels = ['Malignant', 'Benign'], 
            yticklabels = ['Malignant', 'Benign'],
            fmt = 'g',
            cmap = 'Blues')
plt.title('confmat')
plt.xlabel('True Diagnosis')
plt.ylabel('Predicted Diagnosis')
```

![confmat](https://user-images.githubusercontent.com/71438259/209153274-b8f7c81e-77d9-4fa3-9921-5f0607dc3017.jpg)

</br>
Next, we run logistic regression using the sklearn and see how it compares

```
#performing logistic regression using sklearn library
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 42)
logreg.fit(x_train_norm, y_train)

y_pred = logreg.predict(x_test_norm)
confmat2 = confusion_matrix(y_pred, y_test)
confmat2
accuracy_score(y_pred, y_test)

#visualizing confusion matrix
sns.heatmap(confmat2, annot = True, linewidths = 1, 
            xticklabels = ['Malignant', 'Benign'], 
            yticklabels = ['Malignant', 'Benign'],
            fmt = 'g',
            cmap = 'Blues')
plt.title('confmat2')
plt.xlabel('True Diagnosis')
plt.ylabel('Predicted Diagnosis')
```
output:

```
confmat2
array([[105,   3],
       [  3,  60]], dtype=int64)
       
accuracy_score(y_pred, y_test)       
0.9649122807017544
```

![confmat2](https://user-images.githubusercontent.com/71438259/209153469-36181fbd-c81a-4b1f-beb5-e943430420b4.png)

The model we built demonstrates a prediction accuracy of 95.906%, while the one generated from using the sklearn library has a prediction accuracy of 96.49%.
</br>

### Note for deployment of model in clinical settings
For purposes of clinical deployment, it might sometimes be prudent to lower the threshold for prediction from the default value of 0.5 to a lower value to reduce the chance of having false negative results or missed diagnoses. This can be achieved with the following code:

```
#changing default threshold for clinical deployment
y_pred2 = np.where(logreg.predict_proba(x_test_norm)[:, 1] > 0.05, 1, 0)
y_pred2
confmat3 = confusion_matrix(y_pred2, y_test)
confmat3
accuracy_score(y_pred2, y_test)

#visualizing confusion matrix
sns.heatmap(confmat3, annot = True, linewidths = 1, 
            xticklabels = ['Malignant', 'Benign'], 
            yticklabels = ['Malignant', 'Benign'],
            fmt = 'g',
            cmap = 'Blues')
plt.title('confmat3')
plt.xlabel('True Diagnosis')
plt.ylabel('Predicted Diagnosis')
```

output:

```
confmat3
array([[78,  0],
       [30, 63]], dtype=int64)
       
accuracy_score(y_pred2, y_test)
0.8245614035087719
```

![confmat3](https://user-images.githubusercontent.com/71438259/209153648-4eaf69de-f055-447d-b3ff-c98141498d70.png)

Compared to the predictions generated from y_pred, the predictions from y_pred2 are less accurate. An examination of the confusion matrix tells us that the number of false negatives has increased significantly, but the number of false positives is minimized to 0. </br>
If deployed in a clinical setting, this model can potential reduce the doctor's workload by efficiently classifying patients' tumors. 



