Breast Cancer Prediction with K-Nearest Neighbours and Visualization
with Principal Component Analysis
================

## Background

In this project, we will revisit the Wisconsin Breast Cancer dataset. In
a [previous
project](https://github.com/isaaclhk/Projects/blob/main/Python%20projects/breast%20cancer%20prediction.md),
we’ve built a logistic regression model to predict the malignancy of a
breast tumor based on its cell nuclei characteristics. This instance, we
will take a second look at this dataset and visualize the data after
applying principal component analysis (PCA). In addition, we will build
a another prediction model using the K-Nearest Neighbours (KNN)
algorithm.

## About the Dataset

Features are computed from a digitized image of a fine needle aspirate
(FNA) of a breast mass. They describe characteristics of the cell nuclei
present in the image. n the 3-dimensional space is that described in:
\[K. P. Bennett and O. L. Mangasarian: “Robust Linear Programming
Discrimination of Two Linearly Inseparable Sets”, Optimization Methods
and Software 1, 1992, 23-34\].

This database is also available through the UW CS ftp server: ftp
ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on UCI Machine Learning Repository

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

``` r
#load libraries and dataset
library(tidyverse)
library(Hmisc)

data <- read.csv('C:/Users/isaac/OneDrive/Documents/Projects/datasets/data.csv') %>% rename_all(tolower)

#exploratory analysis
head(data)
```
output:
            id diagnosis radius_mean texture_mean perimeter_mean area_mean
    1   842302         M       17.99        10.38         122.80    1001.0
    2   842517         M       20.57        17.77         132.90    1326.0
    3 84300903         M       19.69        21.25         130.00    1203.0
    4 84348301         M       11.42        20.38          77.58     386.1
    5 84358402         M       20.29        14.34         135.10    1297.0
    6   843786         M       12.45        15.70          82.57     477.1
      smoothness_mean compactness_mean concavity_mean concave.points_mean
    1         0.11840          0.27760         0.3001             0.14710
    2         0.08474          0.07864         0.0869             0.07017
    3         0.10960          0.15990         0.1974             0.12790
    4         0.14250          0.28390         0.2414             0.10520
    5         0.10030          0.13280         0.1980             0.10430
    6         0.12780          0.17000         0.1578             0.08089
      symmetry_mean fractal_dimension_mean radius_se texture_se perimeter_se
    1        0.2419                0.07871    1.0950     0.9053        8.589
    2        0.1812                0.05667    0.5435     0.7339        3.398
    3        0.2069                0.05999    0.7456     0.7869        4.585
    4        0.2597                0.09744    0.4956     1.1560        3.445
    5        0.1809                0.05883    0.7572     0.7813        5.438
    6        0.2087                0.07613    0.3345     0.8902        2.217
      area_se smoothness_se compactness_se concavity_se concave.points_se
    1  153.40      0.006399        0.04904      0.05373           0.01587
    2   74.08      0.005225        0.01308      0.01860           0.01340
    3   94.03      0.006150        0.04006      0.03832           0.02058
    4   27.23      0.009110        0.07458      0.05661           0.01867
    5   94.44      0.011490        0.02461      0.05688           0.01885
    6   27.19      0.007510        0.03345      0.03672           0.01137
      symmetry_se fractal_dimension_se radius_worst texture_worst perimeter_worst
    1     0.03003             0.006193        25.38         17.33          184.60
    2     0.01389             0.003532        24.99         23.41          158.80
    3     0.02250             0.004571        23.57         25.53          152.50
    4     0.05963             0.009208        14.91         26.50           98.87
    5     0.01756             0.005115        22.54         16.67          152.20
    6     0.02165             0.005082        15.47         23.75          103.40
      area_worst smoothness_worst compactness_worst concavity_worst
    1     2019.0           0.1622            0.6656          0.7119
    2     1956.0           0.1238            0.1866          0.2416
    3     1709.0           0.1444            0.4245          0.4504
    4      567.7           0.2098            0.8663          0.6869
    5     1575.0           0.1374            0.2050          0.4000
    6      741.6           0.1791            0.5249          0.5355
      concave.points_worst symmetry_worst fractal_dimension_worst  x
    1               0.2654         0.4601                 0.11890 NA
    2               0.1860         0.2750                 0.08902 NA
    3               0.2430         0.3613                 0.08758 NA
    4               0.2575         0.6638                 0.17300 NA
    5               0.1625         0.2364                 0.07678 NA
    6               0.1741         0.3985                 0.12440 NA

``` r
str(data)
```

    'data.frame':   569 obs. of  33 variables:
     $ id                     : int  842302 842517 84300903 84348301 84358402 843786 844359 84458202 844981 84501001 ...
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
     $ x                      : logi  NA NA NA NA NA NA ...

id is dropped because it is not germane to the analysis, ‘x’ is also
dropped because it consists of only null values.

``` r
data = select(data, -c('id', 'x'))
str(data)
```

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

``` r
#exploratory data analysis
describe(data$diagnosis)
```

    data$diagnosis 
           n  missing distinct 
         569        0        2 
                          
    Value          B     M
    Frequency    357   212
    Proportion 0.627 0.373

``` r
barplot(table(data$diagnosis), main = 'Diagnoses')
```

![](breast_cancer_knn_files/figure-commonmark/unnamed-chunk-3-1.png)

There are 357 benign tumors and 212 malignant tumors in the dataset.
Next, the remaining features are described and visualized in a matrix of
histograms. This allows us to inspect the general distribution of each
feature and potentially detect outliers.

``` r
describe(data)
```

    data 

     31  Variables      569  Observations
    --------------------------------------------------------------------------------
    diagnosis 
           n  missing distinct 
         569        0        2 
                          
    Value          B     M
    Frequency    357   212
    Proportion 0.627 0.373
    --------------------------------------------------------------------------------
    radius_mean 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      456        1    14.13    3.848    9.529   10.260 
         .25      .50      .75      .90      .95 
      11.700   13.370   15.780   19.530   20.576 

    lowest : 6.981 7.691 7.729 7.76  8.196, highest: 25.22 25.73 27.22 27.42 28.11
    --------------------------------------------------------------------------------
    texture_mean 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      479        1    19.29    4.789    13.09    14.08 
         .25      .50      .75      .90      .95 
       16.17    18.84    21.80    24.99    27.15 

    lowest : 9.71  10.38 10.72 10.82 10.89, highest: 31.12 32.47 33.56 33.81 39.28
    --------------------------------------------------------------------------------
    perimeter_mean 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      522        1    91.97    26.46    60.50    65.83 
         .25      .50      .75      .90      .95 
       75.17    86.24   104.10   129.10   135.82 

    lowest : 43.79 47.92 47.98 48.34 51.71, highest: 171.5 174.2 182.1 186.9 188.5
    --------------------------------------------------------------------------------
    area_mean 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      539        1    654.9      362    275.8    321.6 
         .25      .50      .75      .90      .95 
       420.3    551.1    782.7   1177.4   1309.8 

    lowest : 143.5 170.4 178.8 181   201.9, highest: 1878  2010  2250  2499  2501 
    --------------------------------------------------------------------------------
    smoothness_mean 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      474        1  0.09636  0.01571  0.07504  0.07965 
         .25      .50      .75      .90      .95 
     0.08637  0.09587  0.10530  0.11482  0.11878 

    lowest : 0.05263 0.06251 0.06429 0.06576 0.06613
    highest: 0.1371  0.1398  0.1425  0.1447  0.1634 
    --------------------------------------------------------------------------------
    compactness_mean 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      537        1   0.1043  0.05694  0.04066  0.04970 
         .25      .50      .75      .90      .95 
     0.06492  0.09263  0.13040  0.17546  0.20870 

    lowest : 0.01938 0.02344 0.0265  0.02675 0.03116
    highest: 0.2832  0.2839  0.2867  0.3114  0.3454 
    --------------------------------------------------------------------------------
    concavity_mean 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      537        1   0.0888  0.08383 0.004983 0.013686 
         .25      .50      .75      .90      .95 
    0.029560 0.061540 0.130700 0.203040 0.243020 

    lowest : 0         0.000692  0.0009737 0.001194  0.001461 
    highest: 0.3635    0.3754    0.4108    0.4264    0.4268   
    --------------------------------------------------------------------------------
    concave.points_mean 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      542        1  0.04892  0.04164 0.005621 0.011158 
         .25      .50      .75      .90      .95 
    0.020310 0.033500 0.074000 0.100420 0.125740 

    lowest : 0        0.001852 0.002404 0.002924 0.002941
    highest: 0.1823   0.1845   0.1878   0.1913   0.2012  
    --------------------------------------------------------------------------------
    symmetry_mean 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      432        1   0.1812  0.03015   0.1415   0.1496 
         .25      .50      .75      .90      .95 
      0.1619   0.1792   0.1957   0.2149   0.2307 

    lowest : 0.106  0.1167 0.1203 0.1215 0.122 , highest: 0.2655 0.2678 0.2743 0.2906 0.304 
    --------------------------------------------------------------------------------
    fractal_dimension_mean 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      499        1   0.0628 0.007536  0.05393  0.05534 
         .25      .50      .75      .90      .95 
     0.05770  0.06154  0.06612  0.07227  0.07609 

    lowest : 0.04996 0.05024 0.05025 0.05044 0.05054
    highest: 0.0898  0.09296 0.09502 0.09575 0.09744
    --------------------------------------------------------------------------------
    radius_se 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      540        1   0.4052   0.2595   0.1601   0.1831 
         .25      .50      .75      .90      .95 
      0.2324   0.3242   0.4789   0.7489   0.9595 

    lowest : 0.1115 0.1144 0.1153 0.1166 0.1186, highest: 1.296  1.37   1.509  2.547  2.873 
    --------------------------------------------------------------------------------
    texture_se 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      519        1    1.217   0.5775   0.5401   0.6404 
         .25      .50      .75      .90      .95 
      0.8339   1.1080   1.4740   1.9094   2.2120 

    lowest : 0.3602 0.3621 0.3628 0.3871 0.3981, highest: 3.12   3.568  3.647  3.896  4.885 
    --------------------------------------------------------------------------------
    perimeter_se 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      533        1    2.866     1.84    1.133    1.280 
         .25      .50      .75      .90      .95 
       1.606    2.287    3.357    5.123    7.042 

    lowest : 0.757  0.7714 0.8439 0.8484 0.873 , highest: 10.05  10.12  11.07  18.65  21.98 
    --------------------------------------------------------------------------------
    area_se 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      528        1    40.34    35.67    11.36    13.16 
         .25      .50      .75      .90      .95 
       17.85    24.53    45.19    91.31   115.80 

    lowest : 6.802 7.228 7.254 7.326 8.205, highest: 199.7 224.1 233   525.6 542.2
    --------------------------------------------------------------------------------
    smoothness_se 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      547        1 0.007041 0.002993 0.003690 0.004224 
         .25      .50      .75      .90      .95 
    0.005169 0.006380 0.008146 0.010410 0.012644 

    lowest : 0.001713 0.002667 0.002826 0.002838 0.002866
    highest: 0.01835  0.02075  0.02177  0.02333  0.03113 
    --------------------------------------------------------------------------------
    compactness_se 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      541        1  0.02548  0.01808 0.007892 0.009169 
         .25      .50      .75      .90      .95 
    0.013080 0.020450 0.032450 0.047602 0.060578 

    lowest : 0.002252 0.003012 0.00371  0.003746 0.00466 
    highest: 0.09586  0.09806  0.1006   0.1064   0.1354  
    --------------------------------------------------------------------------------
    concavity_se 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      533        1  0.03189  0.02625 0.003253 0.007726 
         .25      .50      .75      .90      .95 
    0.015090 0.025890 0.042050 0.058520 0.078936 

    lowest : 0         0.000692  0.0007929 0.0009737 0.001128 
    highest: 0.1435    0.1438    0.1535    0.3038    0.396    
    --------------------------------------------------------------------------------
    concave.points_se 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      507        1   0.0118 0.006495 0.003831 0.005493 
         .25      .50      .75      .90      .95 
    0.007638 0.010930 0.014710 0.018688 0.022884 

    lowest : 0        0.001852 0.002386 0.002404 0.002924
    highest: 0.03441  0.03487  0.03927  0.0409   0.05279 
    --------------------------------------------------------------------------------
    symmetry_se 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      498        1  0.02054 0.008205  0.01176  0.01301 
         .25      .50      .75      .90      .95 
     0.01516  0.01873  0.02348  0.03012  0.03499 

    lowest : 0.007882 0.009539 0.009947 0.01013  0.01029 
    highest: 0.05543  0.05628  0.05963  0.06146  0.07895 
    --------------------------------------------------------------------------------
    fractal_dimension_se 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      545        1 0.003795 0.002343 0.001522 0.001710 
         .25      .50      .75      .90      .95 
    0.002248 0.003187 0.004558 0.006185 0.007960 

    lowest : 0.0008948 0.0009502 0.0009683 0.001002  0.001058 
    highest: 0.01298   0.01792   0.02193   0.02286   0.02984  
    --------------------------------------------------------------------------------
    radius_worst 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      457        1    16.27    5.211    10.53    11.23 
         .25      .50      .75      .90      .95 
       13.01    14.97    18.79    23.68    25.64 

    lowest : 7.93  8.678 8.952 8.964 9.077, highest: 31.01 32.49 33.12 33.13 36.04
    --------------------------------------------------------------------------------
    texture_worst 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      511        1    25.68    6.918    16.57    17.80 
         .25      .50      .75      .90      .95 
       21.08    25.41    29.72    33.65    36.30 

    lowest : 12.02 12.49 12.87 14.1  14.2 , highest: 42.79 44.87 45.41 47.16 49.54
    --------------------------------------------------------------------------------
    perimeter_worst 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      514        1    107.3    36.21    67.86    72.18 
         .25      .50      .75      .90      .95 
       84.11    97.66   125.40   157.74   171.64 

    lowest : 50.41 54.49 56.65 57.17 57.26, highest: 211.7 214   220.8 229.3 251.2
    --------------------------------------------------------------------------------
    area_worst 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      544        1    880.6    568.8    331.1    384.7 
         .25      .50      .75      .90      .95 
       515.3    686.5   1084.0   1673.0   2009.6 

    lowest : 185.2 223.6 240.1 242.2 248  , highest: 3143  3216  3234  3432  4254 
    --------------------------------------------------------------------------------
    smoothness_worst 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      411        1   0.1324  0.02554  0.09573  0.10296 
         .25      .50      .75      .90      .95 
     0.11660  0.13130  0.14600  0.16148  0.17184 

    lowest : 0.07117 0.08125 0.08409 0.08484 0.08567
    highest: 0.1909  0.2006  0.2098  0.2184  0.2226 
    --------------------------------------------------------------------------------
    compactness_worst 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      529        1   0.2543   0.1657  0.07120  0.09368 
         .25      .50      .75      .90      .95 
     0.14720  0.21190  0.33910  0.44784  0.56412 

    lowest : 0.02729 0.03432 0.04327 0.04619 0.04712
    highest: 0.8663  0.8681  0.9327  0.9379  1.058  
    --------------------------------------------------------------------------------
    concavity_worst 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      539        1   0.2722   0.2262  0.01836  0.04565 
         .25      .50      .75      .90      .95 
     0.11450  0.22670  0.38290  0.57132  0.68238 

    lowest : 0        0.001845 0.003581 0.004955 0.005518
    highest: 0.9387   0.9608   1.105    1.17     1.252   
    --------------------------------------------------------------------------------
    concave.points_worst 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      492        1   0.1146  0.07442  0.02429  0.03846 
         .25      .50      .75      .90      .95 
     0.06493  0.09993  0.16140  0.20894  0.23692 

    lowest : 0        0.008772 0.009259 0.01042  0.01111 
    highest: 0.2733   0.2756   0.2867   0.2903   0.291   
    --------------------------------------------------------------------------------
    symmetry_worst 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      500        1   0.2901  0.06473   0.2127   0.2261 
         .25      .50      .75      .90      .95 
      0.2504   0.2822   0.3179   0.3601   0.4062 

    lowest : 0.1565 0.1566 0.1603 0.1648 0.1652, highest: 0.5166 0.544  0.5558 0.5774 0.6638
    --------------------------------------------------------------------------------
    fractal_dimension_worst 
           n  missing distinct     Info     Mean      Gmd      .05      .10 
         569        0      535        1  0.08395  0.01884  0.06256  0.06579 
         .25      .50      .75      .90      .95 
     0.07146  0.08004  0.09208  0.10632  0.11952 

    lowest : 0.05504 0.05521 0.05525 0.05695 0.05737
    highest: 0.1431  0.1446  0.1486  0.173   0.2075 
    --------------------------------------------------------------------------------

``` r
library(ggplot2)
data %>%
  keep(is.numeric) %>%
  gather() %>%
  ggplot(aes(value)) + facet_wrap(~key, scales = 'free') + geom_histogram(bins = 15) + labs(title = 'Summmary of Feature Distributions')
```

![](breast_cancer_knn_files/figure-commonmark/unnamed-chunk-4-1.png)

Separate the features and outcome variable.

``` r
#seperate features and outcome variable
x <- select(data, -diagnosis)
y <- select(data, diagnosis)

names(x)
```

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

``` r
names(y)
```

    [1] "diagnosis"

``` r
y <- unclass(factor(y$diagnosis)) -1
table(y)
```

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

3.  **Project the normalized data onto the principal subspace spanned by
    the eigenvectors of the data covariance matrix with the
    corresponding n largest eigenvalues for a PCA of n components.**

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

### PCA Implementation

Now, we will implement PCA in python code.

``` python
#normalize data
x = r.x
y = r.y

from sklearn.preprocessing import StandardScaler
pca_scaler = StandardScaler()
pca_x = pca_scaler.fit_transform(x)
```

``` python
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca_transformed = pca.fit_transform(pca_x)

pca_x.shape
```

    (569, 30)

``` python
pca_transformed.shape
```

    (569, 2)

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

![](breast_cancer_knn_files/figure-commonmark/unnamed-chunk-8-1.png)

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

## Data Preprocessing

We begin by separating the dataset into training and testing sets. For
this analysis, 70% of the data will be used for training before the
model is tested on the remaining 30%. Random state is set to 42 to
obtain reproducible results.

``` python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, stratify = y, random_state = 42)

x_train.shape
```

    (398, 30)

``` python
x_test.shape
```

    (171, 30)

``` python
len(y_train)
```

    398

``` python
len(y_test)
```

    171

Since the algorithm makes predictions by calculating distances between
data points, we need to scale the data such that all features are
brought to a similar range. This ensures that each feature contributes
proportionally to the distance calculation and avoids bias that may
arise from features having inherently different values or ranges.
Normalizing the data can also mitigate the impact of outliers by
bringing the data within a similar range and reducing the influence of
extreme values.

``` python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
```

    StandardScaler()

``` python
x_train_norm = scaler.transform(x_train)
x_test_norm = scaler.transform(x_test)
```

## KNN Implementation

We perform 10 fold cross validation to determine the optimal number of
neighbors for this model.

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
  
print(cv_scores)
```

    [0.9519871794871795, 0.962051282051282, 0.9595512820512822, 0.964551282051282, 0.9670512820512821, 0.962051282051282, 0.962051282051282, 0.9646153846153845, 0.9570512820512821, 0.9570512820512821, 0.954551282051282, 0.9571794871794872, 0.9571153846153846, 0.9546153846153846, 0.9546153846153846, 0.9521153846153846]

We plot the average accuracy obtained from each set of 10 cross
validations for every ‘k’ against the number of neighbors.

``` python
#plotting cv_scores vs K
plt.close()
sns.lineplot(x = neighbors, y = cv_scores)
plt.title('Average Accuracy Scores vs Neighbors')
plt.show()
```

![](breast_cancer_knn_files/figure-commonmark/unnamed-chunk-12-3.png)

``` python
#calculating optimal number of neighbors
optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print(f'Optimal K =  {optimal_k}')
```

    Optimal K =  9

Based on the above calculation, the optimal number of neighbors is 9. We
shall use this value to fit the final KNN model.

``` python
#fit model
knn = KNeighborsClassifier(n_neighbors = optimal_k)
knn.fit(x_train_norm, y_train)
```

    KNeighborsClassifier(n_neighbors=9)

``` python
y_pred = knn.predict(x_test_norm)
```

Finally, we evaluate the model by calculating the accuracy score and
plotting a confusion matrix.

``` python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print(f'Accuracy: {round(accuracy_score(y_pred, y_test)*100,2)}')
```

    Accuracy: 95.32

``` python
confmat = confusion_matrix(y_pred, y_test)
confmat
```

    array([[106,   7],
           [  1,  57]], dtype=int64)

``` python
print(f'Accuracy: {round(accuracy_score(y_pred, y_test)*100,2)}')
```

    Accuracy: 95.32

``` python
#visualizing confusion matrix
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

![](breast_cancer_knn_files/figure-commonmark/unnamed-chunk-14-5.png)

## References

1.  Deisenroth, M. P. (n.d.) *Mathematics for Machine Learning: PCA*
    \[MOOC\]. Coursera.
    https://www.coursera.org/learn/pca-machine-learning
