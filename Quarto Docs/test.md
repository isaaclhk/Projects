visual test
================

## import and load data

``` python
import pandas as pd
data= pd.read_excel("C:/Users\isaac\OneDrive\Documents\Projects\datasets\cerapro_filling_weight.xlsx")
data.info(verbose = True, show_counts = True)

import seaborn as sns
import matplotlib.pyplot as plt
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 90 entries, 0 to 89
    Data columns (total 3 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   S/N         90 non-null     int64  
     1   FillWeight  90 non-null     float64
     2   Group       90 non-null     object 
    dtypes: float64(1), int64(1), object(1)
    memory usage: 2.2+ KB

## Visualize

``` python
sns.boxplot(data = data, x = 'Group', y = 'FillWeight')
plt.title('Boxplot of FillWeights by Group')
plt.show()

sns.displot(data = data, x = 'FillWeight', hue = 'Group', kind = 'kde', fill = True, alpha = 0.1)
plt.title('kde plot of FillWeights by Group')
plt.show()
```

![](test_files/figure-commonmark/cell-3-output-1.png)

![](test_files/figure-commonmark/cell-3-output-2.png)
