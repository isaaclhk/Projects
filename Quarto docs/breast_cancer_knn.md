Breast Cancer Prediction with K-Nearest Neighbours and Visualization
with Principal Component Analysis
================
Isaac Lam
### Contents
- <a href="#background" id="toc-background">Background</a>
  - <a href="#about-the-dataset" id="toc-about-the-dataset">About the
    Dataset</a>
- <a href="#exploratory-data-analysis"
  id="toc-exploratory-data-analysis">Exploratory Data Analysis</a>
- <a href="#principal-component-analysis-pca"
  id="toc-principal-component-analysis-pca">Principal Component Analysis
  (PCA)</a>
  - <a href="#pre-pca-normalization" id="toc-pre-pca-normalization">Pre-PCA
    Normalization</a>
  - <a href="#pca-implementation" id="toc-pca-implementation">PCA
    Implementation</a>
- <a href="#k-nearest-neighbours-knn"
  id="toc-k-nearest-neighbours-knn">K-Nearest Neighbours (KNN)</a>
  - <a href="#data-preprocessing" id="toc-data-preprocessing">Data
    Preprocessing</a>
  - <a href="#knn-implementation" id="toc-knn-implementation">KNN
    Implementation</a>
- <a href="#references" id="toc-references">References</a>

## Background

In this project, we will revisit the Wisconsin Breast Cancer dataset. In
a [previous
project](https://github.com/isaaclhk/Projects/blob/main/Python%20projects/breast%20cancer%20prediction.md),
we’ve built a logistic regression model to predict the malignancy of a
breast tumor based on its cell nuclei characteristics.

This instance, we will take a second look at this dataset and visualize
the data after applying principal component analysis (PCA). In addition,
we will build a another prediction model using the K-Nearest Neighbours
(KNN) algorithm.

### About the Dataset

Features are computed from a digitized image of a fine needle aspirate
(FNA) of a breast mass. They describe characteristics of the cell nuclei
present in the image. n the 3-dimensional space is that described in:
\[K. P. Bennett and O. L. Mangasarian: “Robust Linear Programming
Discrimination of Two Linearly Inseparable Sets”, Optimization Methods
and Software 1, 1992, 23-34\].

This database is also available through the UW CS ftp server: ftp
ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on the [UCI Machine Learning
Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

Attribute Information:

ID number Diagnosis (M = malignant, B = benign) 3-32) Ten real-valued
features are computed for each cell nucleus:

1)  radius (mean of distances from center to points on the perimeter) b)
    texture (standard deviation of gray-scale values) c) perimeter d)
    area e) smoothness (local variation in radius lengths) f)
    compactness (perimeter^2 / area - 1.0) g) concavity (severity of
    concave portions of the contour) h) concave points (number of
    concave portions of the contour) i) symmetry j) fractal dimension
    (“coastline approximation” - 1)

The mean, standard error and “worst” or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field 13
is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

## Exploratory Data Analysis

We begin by importing relevant libraries and loading the dataset before
describing and visualizing the data.

<details open>
<summary>Code</summary>

``` r
#load libraries and dataset
library(tidyverse)
library(Hmisc)

data <- read.csv('C:/Users/isaac/OneDrive/Documents/Projects/datasets/data.csv') %>% rename_all(tolower)

#exploratory analysis
head(data)
str(data)
```

</details>

id is dropped because it is not germane to the analysis, ‘x’ is also
dropped because it consists of only null values.

<details open>
<summary>Code</summary>

``` r
data = select(data, -c('id', 'x'))
str(data)
```

</details>

    'data.frame':   569 obs. of  31 variables:
     $ diagnosis              : chr  "M" "M" "M" "M" ...
     $ radius_mean            : num  18 20.6 19.7 11.4 20.3 ...
     $ texture_mean           : num  10.4 17.8 21.2 20.4 14.3 ...
     $ perimeter_mean         : num  122.8 132.9 130 77.6 135.1 ...
     $ area_mean              : num  1001 1326 1203 386 1297 ...
     $ smoothness_mean        : num  0.1184 0.0847 0.1096 0.1425 0.1003 ...
     $ compactness_mean       : num  0.2776 0.0786 0.1599 0.2839 0.1328 ...
     $ concavity_mean         : num  0.3001 0.0869 0.1974 0.2414 0.198 ...
     $ concave.points_mean    : num  0.1471 0.0702 0.1279 0.1052 0.1043 ...
     $ symmetry_mean          : num  0.242 0.181 0.207 0.26 0.181 ...
     $ fractal_dimension_mean : num  0.0787 0.0567 0.06 0.0974 0.0588 ...
     $ radius_se              : num  1.095 0.543 0.746 0.496 0.757 ...
     $ texture_se             : num  0.905 0.734 0.787 1.156 0.781 ...
     $ perimeter_se           : num  8.59 3.4 4.58 3.44 5.44 ...
     $ area_se                : num  153.4 74.1 94 27.2 94.4 ...
     $ smoothness_se          : num  0.0064 0.00522 0.00615 0.00911 0.01149 ...
     $ compactness_se         : num  0.049 0.0131 0.0401 0.0746 0.0246 ...
     $ concavity_se           : num  0.0537 0.0186 0.0383 0.0566 0.0569 ...
     $ concave.points_se      : num  0.0159 0.0134 0.0206 0.0187 0.0188 ...
     $ symmetry_se            : num  0.03 0.0139 0.0225 0.0596 0.0176 ...
     $ fractal_dimension_se   : num  0.00619 0.00353 0.00457 0.00921 0.00511 ...
     $ radius_worst           : num  25.4 25 23.6 14.9 22.5 ...
     $ texture_worst          : num  17.3 23.4 25.5 26.5 16.7 ...
     $ perimeter_worst        : num  184.6 158.8 152.5 98.9 152.2 ...
     $ area_worst             : num  2019 1956 1709 568 1575 ...
     $ smoothness_worst       : num  0.162 0.124 0.144 0.21 0.137 ...
     $ compactness_worst      : num  0.666 0.187 0.424 0.866 0.205 ...
     $ concavity_worst        : num  0.712 0.242 0.45 0.687 0.4 ...
     $ concave.points_worst   : num  0.265 0.186 0.243 0.258 0.163 ...
     $ symmetry_worst         : num  0.46 0.275 0.361 0.664 0.236 ...
     $ fractal_dimension_worst: num  0.1189 0.089 0.0876 0.173 0.0768 ...

We observe the distribution of malignant and benign tumors in this
dataset.

<details open>
<summary>Code</summary>

``` r
#exploratory data analysis
describe(data$diagnosis)
```

</details>

    data$diagnosis 
           n  missing distinct 
         569        0        2 
                          
    Value          B     M
    Frequency    357   212
    Proportion 0.627 0.373

<details open>
<summary>Code</summary>

``` r
barplot(table(data$diagnosis), main = 'Diagnoses')
```

</details>

<img src = 'https://github.com/isaaclhk/Projects/assets/71438259/308833f4-aaed-4dbb-a0bf-8b59b4b3c52f' width = 500>

There are 357 benign tumors and 212 malignant tumors in the dataset.
Next, the remaining features are described and visualized in a matrix of
histograms. This allows us to inspect the general distribution of each
feature and potentially detect outliers.

<details open>
<summary>Code</summary>

``` r
describe(data)
```

</details>
<details open>
<summary>Code</summary>

``` r
library(ggplot2)
data %>%
  keep(is.numeric) %>%
  gather() %>%
  ggplot(aes(value)) + facet_wrap(~key, scales = 'free') + geom_histogram(bins = 15) + labs(title = 'Summmary of Feature Distributions')
```

</details>

<img src= https://github.com/isaaclhk/Projects/assets/71438259/7a09ca67-cfbb-4ed5-a6b1-f761cd32d959 width = 50%>
![unnamed-chunk-5-1](https://github.com/isaaclhk/Projects/assets/71438259/7a09ca67-cfbb-4ed5-a6b1-f761cd32d959)

Here we separate the features and outcome variable for analysis later.

<details open>
<summary>Code</summary>

``` r
#seperate features and outcome variable
x <- select(data, -diagnosis)
y <- select(data, diagnosis)
```

</details>
<details open>
<summary>Code</summary>

``` r
#check features
names(x)
```

</details>

     [1] "radius_mean"             "texture_mean"           
     [3] "perimeter_mean"          "area_mean"              
     [5] "smoothness_mean"         "compactness_mean"       
     [7] "concavity_mean"          "concave.points_mean"    
     [9] "symmetry_mean"           "fractal_dimension_mean" 
    [11] "radius_se"               "texture_se"             
    [13] "perimeter_se"            "area_se"                
    [15] "smoothness_se"           "compactness_se"         
    [17] "concavity_se"            "concave.points_se"      
    [19] "symmetry_se"             "fractal_dimension_se"   
    [21] "radius_worst"            "texture_worst"          
    [23] "perimeter_worst"         "area_worst"             
    [25] "smoothness_worst"        "compactness_worst"      
    [27] "concavity_worst"         "concave.points_worst"   
    [29] "symmetry_worst"          "fractal_dimension_worst"

<details open>
<summary>Code</summary>

``` r
#check outcome
names(y)
```

</details>

    [1] "diagnosis"

<details open>
<summary>Code</summary>

``` r
#Change B and M to 0 and 1
y <- unclass(factor(y$diagnosis)) -1
table(y)
```

</details>

    y
      0   1 
    357 212 

## Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a dimensionality reduction
technique used to transform a high-dimensional dataset into a
lower-dimensional representation while preserving as much variation of
the feature set as possible. This enables us to visualize or explore the
classification power of a high-dimensional dataset.

The PCA can generally be computed in the following 3 steps:

**1. Normalize the data**

For this analysis, we will use z-score normalization which transforms
each feature to have a mean of 0 and standard deviation of 1. The
formula for z-score normalization is shown below:

$$
{\LARGE z = \frac{x-u}{\sigma}}
$$ **2. Compute the data covariance matrix**

$$
{\LARGE Cov(x,y) = \frac{\sum(x_{i}- \bar{x})*(y_{i}-\bar{y})}{N}}
$$

**3. Project the normalized data onto the principal subspace spanned by
the eigenvectors of the data covariance matrix with the corresponding n
largest eigenvalues for a PCA of n components.**

    This projection can be described as:

$$
{\LARGE \tilde x* = \pi_{u}(x*) = BB^Tx*}
$$

Where $x*$ refers to $x$ normalized, $pi_{u}$ refers to the projection
of $x*$ onto the principal subspace $u$, and B is the matrix that
contains the eigenvectors that belong to the largest eigenvalues as
columns, then $B^Tx*$ are the coordinates of the projection with respect
to the basis of the principal subspace.

Further details on the derivation of PCA are covered in [this
course](https://www.coursera.org/learn/pca-machine-learning)
(Deisenroth., n.d.).

Now, we will implement PCA in python code.

### Pre-PCA Normalization

<details open>
<summary>Code</summary>

``` python
#normalize data
x = r.x
y = r.y

from sklearn.preprocessing import StandardScaler
pca_scaler = StandardScaler()
pca_x = pca_scaler.fit_transform(x)
```

</details>

### PCA Implementation

<details open>
<summary>Code</summary>

``` python
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca_transformed = pca.fit_transform(pca_x)

pca_x.shape
```

</details>

    (569, 30)

<details open>
<summary>Code</summary>

``` python
pca_transformed.shape
```

</details>

    (569, 2)

<details open>
<summary>Code</summary>

``` python
import seaborn as sns
import matplotlib.pyplot as plt
plt.close()

sns.set_style('darkgrid')
pca_plot = sns.scatterplot(x = pca_transformed[:, 0], y = pca_transformed[:, 1], hue = y)
plt.legend(title = 'Tumor Classification', labels = ['Malignant', 'Benign'])
plt.title('Dimensional Reduction of Breast Cancer Dataset to 2 Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.show(pca_plot)
```

</details>

![unnamed-chunk-10-1](https://github.com/isaaclhk/Projects/assets/71438259/61bfb36d-542c-4c0d-a147-09207c4dad6f)

PCA projects data points onto the lower-dimensional space spanned by the
principal components. By visualizing the projected data, we gain a
better understanding of the relationships and patterns within the
dataset. For example, in the figure above, we see a distinct separation
between malignant and benign tumors. This segregation indicates that the
data has high predictive strength. It should be noted, however, that the
employed PCA technique captures the maximum variance realizable within
two dimensions, thus leaving some unaccounted variance from additional
dimensions. Consequently, certain benign tumor data points appear within
the cluster of malignant tumor data points and vice versa. But this does
not necessarily mean that they will be inaccurately predicted,as a
predictive model trained on the complete set of features might yield
more precise predictions.

## K-Nearest Neighbours (KNN)

KNN is a non-parametric supervised machine learning algorithm. The basic
idea behind the KNN algorithm is to classify a new data point or predict
its value based on its proximity to its neighboring data points in the
feature space. In other words, it assumes that data points with similar
features tend to belong to the same class or have similar output values.

The default method used by sklearn to calculate distance is the
Minkowski Distance, which is a generalization of the euclidean distance
in ‘c’ dimensions. The formula for Minkowski distance is:

$$
{\LARGE d(x,y) = (\sum_{i=1}^n \vert xi - yi\vert^c)^\frac{1}{c}}
$$

Once the pre-specified ‘k’ number of nearest neighbouring data points
are identified, a voting mechanism is used to determine the class label
for the new data point. Each neighbor gets to vote, and the majority
class among the K neighbors is assigned as the predicted class for the
new data point. For example, if K = 5 and K nearest neighbours of a new
data point are labelled ‘M’, ‘B’, ‘M’, ‘M’, ‘B’, the KNN algorithm
assigns the class with the majority votes, which is ‘M’, to the new data
point.

### Data Preprocessing

We begin by separating the dataset into training and testing sets. For
this analysis, 70% of the data will be used for training before the
model is tested on the remaining 30%. Random state is set to 42 to
obtain reproducible results.

<details open>
<summary>Code</summary>

``` python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, stratify = y, random_state = 42)

x_train.shape
```

</details>

    (398, 30)

<details open>
<summary>Code</summary>

``` python
x_test.shape
```

</details>

    (171, 30)

<details open>
<summary>Code</summary>

``` python
len(y_train)
```

</details>

    398

<details open>
<summary>Code</summary>

``` python
len(y_test)
```

</details>

    171

Since the algorithm makes predictions by calculating distances between
data points, we need to scale the data such that all features are
brought to a similar range. This ensures that each feature contributes
proportionally to the distance calculation and avoids bias that may
arise from features having inherently different values or ranges.
Normalizing the data can also mitigate the impact of outliers by
bringing the data within a similar range and reducing the influence of
extreme values.

<details open>
<summary>Code</summary>

``` python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
```

</details>

    StandardScaler()

<details open>
<summary>Code</summary>

``` python
x_train_norm = scaler.transform(x_train)
x_test_norm = scaler.transform(x_test)
```

</details>

### KNN Implementation

We perform 10 fold cross validation to determine the optimal number of
neighbors for this model.

<details open>
<summary>Code</summary>

``` python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

neighbors = []
cv_scores = []

from sklearn.model_selection import cross_val_score
# Perform 10 fold cross validation
for k in range(1, 32, 2):
  neighbors.append(k)
  knn = KNeighborsClassifier(n_neighbors = k)
  scores = cross_val_score(knn, x_train_norm, y_train, cv = 10, scoring = 'accuracy')
  cv_scores.append(np.mean(scores))
```

</details>

We plot the average accuracy obtained from each set of 10 cross
validations for every ‘k’ against the number of neighbors.

<details open>
<summary>Code</summary>

``` python
#plotting cv_scores vs K
plt.close()
sns.lineplot(x = neighbors, y = cv_scores)
plt.title('Average Accuracy Scores vs Neighbors')
plt.show()
```

</details>

![unnamed-chunk-14-3](https://github.com/isaaclhk/Projects/assets/71438259/9a4cf34b-ed85-44ba-ac50-eda4843fb084)

<details open>
<summary>Code</summary>

``` python
#calculating optimal number of neighbors
optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print(f'Optimal K =  {optimal_k}')
```

</details>

    Optimal K =  9

Based on the above calculation, the optimal number of neighbors is 9. We
shall use this value to fit the final KNN model.

<details open>
<summary>Code</summary>

``` python
#fit model
knn = KNeighborsClassifier(n_neighbors = optimal_k)
knn.fit(x_train_norm, y_train)
```

</details>

    KNeighborsClassifier(n_neighbors=9)

<details open>
<summary>Code</summary>

``` python
y_pred = knn.predict(x_test_norm)
```

</details>

Finally, we evaluate the model by calculating the accuracy score and
plotting a confusion matrix.

<details open>
<summary>Code</summary>

``` python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print(f'Our model has an accuracy score of: {round(accuracy_score(y_pred, y_test)*100,2)}')
```

</details>

    Our model has an accuracy score of: 95.32

<details open>
<summary>Code</summary>

``` python
#visualizing confusion matrix
confmat = confusion_matrix(y_pred, y_test)
plt.close()
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(confmat, annot = True, linewidths = 1, 
            xticklabels = ['Benign', 'Malignant'], 
            yticklabels = ['Benign', 'Malignant'],
            fmt = 'g',
            cmap = 'Blues')
plt.title('confmat')
plt.xlabel('True Diagnosis')
plt.ylabel('Predicted Diagnosis')
plt.show()
```

</details>

![unnamed-chunk-16-5](https://github.com/isaaclhk/Projects/assets/71438259/0c507a15-607e-471b-a4da-2423eaf3fabe)


## References

1.  Deisenroth, M. P. (n.d.) *Mathematics for Machine Learning: PCA*
    \[MOOC\]. Coursera.
    https://www.coursera.org/learn/pca-machine-learning
2.  Wolberg, W H., Street, W N., Mangasarian, O L. (1995, November)
    Breast Cancer Winconsin (Diagnostic) Data set, Retrieved from
    https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data.
