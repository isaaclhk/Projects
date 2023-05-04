import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('C:/Users/isaac/OneDrive/Documents/Projects/datasets/data.csv')
print(df.head())

df.info()

#drop Unnamed 32 because all nulls
df.drop(['Unnamed: 32', 'id'], axis = 1, inplace = True)

#check that the null column was dropped, note that all are floats except diagnosis
df.info()
df.shape

#examine diagnosis variable
df['diagnosis'].unique()
df['diagnosis'].value_counts()
#M = malignant, B = benign, convert to 1 = malignant, 0 = benign
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})
df['diagnosis'].value_counts()

#visualize malignant vs benign
sns.countplot(x = 'diagnosis', data = df)
plt.title('Malignant vs Benign')
plt.show()

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

#checking for multicollinearity
corr = df.corr().round(2)
#creating triangular heatmap
mask = np.triu(np.ones_like(corr, dtype = np.bool))
sns.heatmap(corr, annot = True, mask = mask, xticklabels = True, yticklabels = True)

#multicollinear variables
'''
worsts and means are collinear (drop worsts as it is a subset of means, unable to tell which will be the stronger predictor based on plots)
radius, perimeter and area are collinear (all are measures of cell size, retain area as it is the most intuitive measure of cell size)
compactness, concavity and concave points are collinear(all are measure of cell shape. 
                                                        retain compactness  as it is a measure of overall shape)

if time isnt a constraint, can test every variable. but for the purposes of this project,
we'll drop collinear variables.
'''

#drop worst
df.drop(df.loc[:,'radius_worst': 'fractal_dimension_worst'], axis = 1, inplace = True)
df.columns

#drop radius and perimeter
df.drop(['area_mean', 'perimeter_mean', 'area_se', 'perimeter_se'], axis = 1, inplace = True)
df.columns

#drop concavity and concave points
df.drop(['concavity_mean', 'concave points_mean', 'concavity_se', 'concave points_se'], axis = 1, inplace = True)
df.columns

#replot correlation matrix
corr = df.corr().round(2)
mask = np.triu(np.ones_like(corr, dtype = np.bool))
sns.heatmap(corr, annot = True, mask = mask, xticklabels = True, yticklabels = True)

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

#sigmoid function
def sigmoid(z):
    g = 1/(1 + np.exp(-z))
    return g


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
    
#predict
def predict(X, w, b):
    
    m,n = X.shape
    p = np.zeros(m)
    
    for i in range(m):
        f_wb_i = sigmoid(np.dot(w, X.iloc[i]) + b)
        p[i] = f_wb_i >= 0.5
    
    return p

#fitting the model
initial_w = np.zeros(x_train_norm.shape[1])
initial_b = 0
iterations = 1000
alpha = 0.05


w, b, J_history, i_history = gradient_descent(x_train_norm, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print(w)
print(b)

#plot cost against interations
sns.lineplot(x= i_history, y= J_history)
plt.title('Gradient Descent')
plt.xlabel('Iterations')
plt.ylabel('Total cost')

#making prediction
p = predict(x_test_norm, w, b)

#evaluating accuracy
print(f'Accuracy: {np.mean(p == y_test) * 100}')
from sklearn.metrics import confusion_matrix, accuracy_score
confmat = confusion_matrix(p, y_test)
confmat

#visualizing confusion matrix
sns.heatmap(confmat, annot = True, linewidths = 1, 
            xticklabels = ['Benign', 'Malignant'], 
            yticklabels = ['Benign', 'Malignant'],
            fmt = 'g',
            cmap = 'Blues')
plt.title('confmat')
plt.xlabel('True Diagnosis')
plt.ylabel('Predicted Diagnosis')

    
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
            xticklabels = ['Benign', 'Malignant'], 
            yticklabels = ['Benign', 'Malignant'],
            fmt = 'g',
            cmap = 'Blues')
plt.title('confmat2')
plt.xlabel('True Diagnosis')
plt.ylabel('Predicted Diagnosis')

#changing default threshold for clinical deployment
y_pred2 = np.where(logreg.predict_proba(x_test_norm)[:, 1] > 0.05, 1, 0)
y_pred2
confmat3 = confusion_matrix(y_pred2, y_test)
confmat3
accuracy_score(y_pred2, y_test)

#visualizing confusion matrix
sns.heatmap(confmat3, annot = True, linewidths = 1, 
            xticklabels = ['Benign', 'Malignant'], 
            yticklabels = ['Benign', 'Malignant'],
            fmt = 'g',
            cmap = 'Blues')
plt.title('confmat3')
plt.xlabel('True Diagnosis')
plt.ylabel('Predicted Diagnosis')




#xgboost
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(x_train, y_train)
y_pred3 = model.predict(x_test)
accuracy_score(y_pred3, y_test)
confmat4 = confusion_matrix(y_pred3, y_test)
confmat4










