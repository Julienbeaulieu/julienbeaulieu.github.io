# [Kaggle competition: IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/overview) 

Can we detect fraud from customer transactions? Lessons from my first competition.

Link to the notebook [on GitHub](https://github.com/Julienbeaulieu/fraud-detection-kaggle-competition).

# 1. Motivation


After having spent a lot of time taking data science classes, I was eager to start practicing on a real dataset and to enter a Kaggle competition. I am thankful that I did because in the process, I learned a lot of things that aren't covered in those classes. Techniques like stratified cross validation, increasing the memory efficiency of my dataset, model stacking and blending, were all new to me.   

I also applied techniques learnt in Fastai's Intro to machine learning course which I'll comment on throughout the notebook. I highly recommend this course if you are learning like me. 

Even though my ranking was nothing impressive (top 45%), I now understand what it takes to create a state of the art kernel and have learned the tools to do so intelligently and efficiently. 

I am sharing my solution, methodology and a bunch of efficient helper functions as a way to anchor these learnings and for beginners who want to get better.

Topics covered in the notebook:
- Fastai helper functions to clean and numericalize a data set.
- A function to reduce the memory usage of your DataFrame.
- A methodology to quickly run a model to gain a better understanding of our dataset
- How Exploratory Data Analysis informs our feature engineering decisions
- Feature selection using LGBM's ` feature_importance` attribute.
- Crossvalidation for TimeSeriesSplit and StratifiedKFold with LGBM
- The code and helper functions for stacking and ensembling models
- Fastai tips and tricks throughout the notebook

# 2. About this dataset

In this competition we are predicting the probability that an online transaction is fraudulent, as denoted by the binary target `isFraud`. The data comes from [Vesta Corporation's](https://trustvesta.com/) real-world e-commerce transactions and contains a wide range of features from device type to product features.

The data is broken into two files: identity and transaction, which are joined by `TransactionID`.

> Note: Not all transactions have corresponding identity information.



**Evaluation**

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.


# 3. Methodology

**1. Quick cleaning and modeling**.

When starting off with a dataset with as many columns as this one (over 400), we first run it through an ensemble learner - LGBM - forgoing any exploratory data analysis and feature engineering at the beginning. Once the model is fit to the data, we have a look at the features which are the most important using LGBM's feature_importances attribute. 
This allows us to concentrate our efforts on only the most important features instead of spending time looking at features with little to no predictive power. 

**2. Understand the data with EDA**.

Once we've filtered our columns, we'll look at the ones with the highest importance. The findings in this analysis will guide our feature engineering efforts in the next section. Some questions we'll want to answer:
- How are the top features related to our target variable? 
- What are their distributions like if we plot them with histograms and countplots?  
- What's their relationship with other important features? Do they seem to be related?
- Are there any features that we can split into multiple columns or simplify in any way?
- etc.

**3. Feature engineering**.

Once we understand our data, we can start creating new columns by splitting up current ones, transforming them to change their scale or looking at their mean, combining new ones to create interactions, and much more. 

**4. Train different models, fit them to the training data with cross validation, and perform model stacking and/or blending**.

The models I tested were RandomForests, XGBoost, and LightGBM. 

I tried several cross validation techniques such as Stratification and TimeSeriesSplit, neither of which beat my single model LGBM, but it was a great learning experience to code it. 
I also discovered several powerful ensemble techniques which are used by top Kaggle contenders: stacking, blending, averaging our least correlated submissions, etc.  

**Here is the high-level summary of the scores of my different Kaggle competition submissions:**

- Base features with stock LGBM: 0.87236
- All engineered features with stock : 0.89821 (+0.02585)
- All engineered features with LGBM hyperparameter tuning: 0.91394 (+0.02176)
- Add feature selection: 0.91401 (+0.0007) - **This was my best score. Rank: 2916/6438**
- Use TimeSeriesSplit crossvalidation: 0.89926 (-0.01476 vs top score)
- Use Stratified crossvalidation: 0.90473 (-0.00929 vs top score)
- Stack 3 tree base models for level 1, and use a LGBM for level 2: 0.85597 (-0.05804 vs top score) (base models weren't optimized as much as I wanted, which explains the poor performance)
- Use a weighted average on my top submissions: 0.91145 (-0.00256) - (I think I could have optmized this and gotten a better score)

---
📣**Insights**

> Note: I'll be supplementing this notebook with an 📣**Insights** section where I share techniques I learned from Fastai course as well best practices collected from reading Kaggle discussions and kernels. 


---

# 4. Importing the libraries and reading the dataset


```python
%load_ext autoreload
%autoreload 2

%matplotlib inline
```
   


```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import forest
import os
import re
import feather
from pandas import get_dummies
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy as hc
import scipy
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, auc
```

## 4.1 Loading the data

---
📣 **Insights**:
- Putting a exclamation point at the beginning of a cell allows us to write command line code. I used this to install new libraries when needed straight from the notebook (such as feather) and for other useful commands like `ls`.
- The following function `display_all` is copy-pasted from Fastai. Very practical for when we're visually assessing our dataframes since it allows us to view as many columns and rows of a DataFrame as we want. I used it all the time throughout this project.



```python
# Path to our data folder
PATH = "data/"

# Show the contents of our data folder
!ls "data"
```

    submission_logs.txt
    test_identity.csv
    test_transaction.csv
    train_identity.csv
    train_transaction.csv
    


```python
# load training_set
df_id_train = pd.read_csv(f'{PATH}train_identity.csv')
df_trans_train = pd.read_csv(f'{PATH}train_transaction.csv')

# load test_set
df_id_test = pd.read_csv(f'{PATH}test_identity.csv')
df_trans_test = pd.read_csv(f'{PATH}test_transaction.csv')
```


```python
# Smaller helped function for visualization
def display_all(df):
    '''
    Small helper function to allow us to disaply 1000 rows and columns. This will come in handy 
    because we are dealing with a lot of columns
    '''
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
```

## 4.2 Merge our dataframes


```python
df_train = df_trans_train.merge(df_id_train, on='TransactionID', how='left')
df_test = df_trans_test.merge(df_id_test, on='TransactionID', how='left')
```


```python
# Set 'TransactionID' as index
df_train = df_train.set_index('TransactionID')
df_test = df_test.set_index('TransactionID')
```


```python
# Create a copy of the original datasets - ones without any changes
df_train_og = df_train.copy()
df_test_og = df_test.copy()
```


```python
# Sanity check for the merge
df_train.shape, df_test.shape
```




    ((590540, 433), (506691, 432))



# 5. Quick Cleaning and Modeling

## 5.1 Convert strings to pandas categories

A lot of our variables are currently stored as strings, which is inefficient, and doesn't provide the numeric coding required to run our models. Therefore, we create `train_cats` function to convert strings to pandas categories. This will also allow us to OneHotEncode some of these categories later on. 

**Important note regarding categories**: We need to make sure that the order of our categories are the same for both our training and testing set. Category codes in our training set may differ from our test set if we simply apply `train_cats` to both.  

**Solution**: We'll create a function `apply_cats` which does the same thing as `train_cats` but will additionally use `df_train` as a template for the category codes. This ensures the order of our categories in both DataFrames are the same.  

---
📣**Insights**: 

`train_cats` and `apply_cats` are very useful functions taken from Fastai. Using them allows us to use all of the columns of our dataset, instead of having to discard them because they wouldn't be in the right format. We therefore have more information to work with. 


```python
def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
```


```python
def apply_cats(df):
    for n,c in df.items():
        if (n in df_train.columns) and (df_train[n].dtype.name=='category'):
            df[n] = c.astype('category').cat.as_ordered() # same code as train_cats(df)
            df[n].cat.set_categories(df_train[n].cat.categories, ordered=True, inplace=True) # Use df_train as a template
```


```python
train_cats(df_train)
apply_cats(df_test)
```

## 5.2 Clean NaN values, apply OneHotEncoding and create new columns based on the NaN values

We'll apply the steps in the title using functions. These were taken from Fastai and slighty simplified for our purposes. Let's review them one by one. 
1. `numericalize`: Changes a categorical type column from text to its integer codes so that it can be used by our model.
2. `fix_missing`: Impute missing data in a column of `df` with the median, and add a `{col_name}_na` column related to the NaN values. The column will show a `0` for rows that didn't have NaN values, and show a `1` if the data was missing.
3. `proc_df`: Takes a data frame, splits off the response variable (y), and changes the df into an entirely numeric dataframe by calling `numericalize()` which converts the category columns to their matching category codes. For each column of df which is not in skip_flds nor in ignore_flds, NaN values are replaced by the median value of the column (using `fix_missing()`).
    
    Returns: `[x, y, nas]` 
    - `x`: x is the transformed version of df. x will not have the response variable and is entirely numeric.
    - `y`: y is the response variable
    - `nas`: returns a dictionary of which NaNs it created, and the associated median.


```python
def numericalize(df, col, name, max_n_cat):
    '''
    Details: If the column is not numeric, AND if max_n_cat is not specified OR if the number of categories 
             in the columns is <= max_n_cat, then we replace the column by its category codes
    '''
    if not is_numeric_dtype(col) and (max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = pd.Categorical(col).codes+1
```


```python
def fix_missing(df, col, name, na_dict):
    '''
    Details: If the column has null values or if we passed in a na_dict:
             Then we create a new column [name+'_na'] indicating where the NaNs were
             
    '''
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict
```


```python
def proc_df(df, y_fld=None, na_dict=None, max_n_cat=None):   
    
    df = df.copy()
    
    if y_fld is None:
        y = None
        y_fld = []
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
    df.drop(y_fld, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    
    # Call fix_missing() to replace NaN values by the median, and create new NaN columns
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
        
    # Apply numericalize() to change a column to it's category code
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True) # get_dummie checks for everything that is still a category and OneHotEncodes
    
    res = [df, y, na_dict]

    return res
```

Let's call `proc_df` and pass the argument `max_n_cat=8` which will apply `get_dummies()` on category types that have 8 categories or less.


```python
# Apply proc_df to transform our dataframe, remove NAs and create new columns with NAs
df_train, y, nas = proc_df(df_train, 'isFraud', max_n_cat=8)
```


```python
# Apply proc_df to the test dataframe
df_test, _, nas = proc_df(df_test, na_dict=nas, max_n_cat=8)
```


```python
# Visual assessment of the result
# display_all(df_train)
```

---
📣**Insights**: 

`proc_df`, `numericalize` and `fix_missing` allow us to use all columns in our dataframe for our model. We're also getting extra columns with potentially very good information: for each column that contained a NaN value, we are creating a new function telling us where the NaN value was. If these columns end up being useless, they will get cleaned afterwards. 


With just very little effort, our dataframe is already ready to for training.

## 5.3 Training and creating a validation set with timeseries data

---
📣**Insights**: 

1. In general, if we are dealing with timeseries dataset like this one, we want our validation set to be from a different time period than our training set. This is because we want to evaluate whether our model is good at prediction the future. For example, if we look at Kaggle's test dataset, the time periods are indeed different. This can be seen in the values of `TransactionDT` feature.   
Therefore, we create the validation set with a sequential set of rows instead of picking random samples of our training set.  


2. For large datasets, when finding the best hyperparameters and deciding which features to keep, we should first work on a sample of our dataframe so that the training time is reduced. This allows us to test more at a fraction of the time. Only once we think we have a good model should we can try it on the whole data set. 

**Training and validation set function on all the data**


```python
# Create a function that will split our training dataset. 
def split_vals(a, n):
    return a[:n].copy(), a[n:].copy()
```

**Sampling training and validation set function**


```python
# Create a function that will create a sample training and validation set 
def split_vals_sample(a, n, m):
    return a[500000:500000+n].copy(), a[500000+n:500000+m].copy()
```


```python
# Choose a large enough n
n = 500_000
X_train, X_valid = split_vals(df_train, n)
y_train, y_valid = split_vals(y, n)
raw_train, raw_valid = split_vals(df_train_og, n)

print(F'X_train: {len(X_train)}, X_valid: {len(X_valid)}, y_train: {len(y_train)}, y_valid: {len(y_valid)}')
```

    X_train: 500000, X_valid: 90540, y_train: 500000, y_valid: 90540
    

# 6. Base LGBM model and feature importance


```python
params = {'metric': 'auc'}

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_valid, label=y_valid)

%time clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)
```

    Training until validation scores don't improve for 500 rounds.
    [200]	training's auc: 0.927545	valid_1's auc: 0.879584
    [400]	training's auc: 0.943789	valid_1's auc: 0.885718
    [600]	training's auc: 0.953669	valid_1's auc: 0.891369
    [800]	training's auc: 0.961838	valid_1's auc: 0.891834
    [1000]	training's auc: 0.966789	valid_1's auc: 0.891689
    Early stopping, best iteration is:
    [653]	training's auc: 0.955934	valid_1's auc: 0.89347
    Wall time: 4min 27s
    

## 6.1 Model interpretation
Without doing much we don't get a great score. But what we're interested in are the features that have the most importance so focus on them and get more insights. 

Let's use LightGBM's plot_importance method to plot the top 50 features that have the most impact.


```python
lgb.plot_importance(clf, figsize=(15,20), max_num_features=50)
```



![png](https://julienbeaulieu.github.io/public/fraud-detection-output/output_45_1.png)


From the feature importance plot, we know that the following features are important and worth looking at in detail:
- All card features (`card1`, `card2`, etc.): payment card information, such as card type, card category, issue bank, country, etc.
- `TransactionDT`: timedelta from a given reference datetime (not an actual timestamp)
- `TransactionAmt`: transaction payment amount in USD
- `addr1`: address
- `P_emaildomain`, `R_emaildomain`: purchaser and recipient email domain
- `C13`, `C1`, `C2`, `C14`: related with counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
- `id_31`: Information about the user's browser.

There are more but these are the main ones we'll look at.

# 7. Exploratory Data Analysis

> Note: I will not go very deep into EDA since this is not the focus of this notebook. I will provide a few examples as to how EDA prompted feature engineering decision that were made on the dataset as well as some basic visualizations.  

I recognize that in a real world scenario, a big chunk of the effort would be necessary for this stage. It is incredibly important to fully understand the data we are working with, and to work with different subject matter experts in the organization to validate our findings and to push our understanding further. I save this for when I'll try to win a competition ;)     

---
📣**Insights**: 

When dealing with a lot of data, it can be hard to visualize all the points on a graph. We can create a sampling function to deal with this issue. 



```python
def get_sample(df,n):
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()

df_sample = get_sample(df_train, 500)
```


```python
# Add 'isFraud' back to our dataframe for EDA purposes
df_train['isFraud'] = df_train_og['isFraud']
```

## 7.1 Target variable: IsFraud

Only **20,663** or **3.5%** of transactions were fraudulent. 


```python
plt.figure(figsize=(10, 3))
sns.countplot(data=df_train, y= 'isFraud')
```








![png](https://julienbeaulieu.github.io/public/fraud-detection-output/output_53_1.png)



```python
df_train.isFraud.value_counts()
```




    0    569877
    1     20663
    Name: isFraud, dtype: int64



## 7.2 Cards

Let's start by visualizing the relationships and distributions of our numerical Cards variables.


```python
sns.pairplot(df_sample[['card1', 'card2', 'card3', 'card5']])
```








![png](https://julienbeaulieu.github.io/public/fraud-detection-output/output_57_1.png)


Not much can be interpreted other than 'card3' and 'card5' are dominated by one value.

**card4 & card5:**


```python
plt.figure(figsize=[12,5])

# card4
plt.subplot(1,2,1)
sns.countplot(data=df_train, x='card4')
plt.title('Card4 Variable')

# card6
plt.subplot(1,2,2)
sns.countplot(data=df_train, x='card6')
plt.title('Card6 Variable')
```








![png](https://julienbeaulieu.github.io/public/fraud-detection-output/output_59_1.png)


These variables will definitely be One Hot Encoded to make sure we're capturing all of their influence. 

## 7.3 TransactionAMT 

As we'll see below, taking the log of the transaction amount really helps interpretation.

### Distribution


```python
# Take the log transaction amount with plt.xscale('log') and adjust the bins and ticks accordingly
data = df_train.TransactionAmt
data_bins = 10 ** np.arange(0.2, np.log10(data.max())+0.2, 0.2)
plt.hist(data, bins=data_bins);
plt.xscale('log')
tick_loc = [5, 10, 30, 100, 300, 1000, 3000, 10000]
plt.xticks(tick_loc, tick_loc)
plt.xlabel('log values');
```


![png](https://julienbeaulieu.github.io/public/fraud-detection-output/output_64_0.png)


### Distribution of fraudulent VS non fraudulent transaction amounts


```python
plt.figure(figsize=[5, 10])

plt.subplot(2,1,1)
data = df_train[df_train['isFraud'] == 1].TransactionAmt
log_bins = 10 ** np.arange(0.2, np.log10(data.max())+0.2, 0.2)
plt.hist(data, bins=log_bins);
plt.xscale('log')
tick_loc = [5, 10, 30, 100, 300, 1000, 3000, 10000, 38000]
plt.xticks(tick_loc, tick_loc)
plt.xlabel('Amount in USD');
plt.title('Transaction is Fraudulent');

plt.subplot(2,1,2)
data = df_train[df_train['isFraud'] == 0].TransactionAmt
log_bins = 10 ** np.arange(0.2, np.log10(data.max())+0.2, 0.2)
plt.hist(data, bins=log_bins);
plt.xscale('log')
tick_loc = [5, 10, 30, 100, 300, 1000, 3000, 10000, 38000]
plt.xticks(tick_loc, tick_loc)
plt.xlabel('Amount in USD');
plt.title('Transaction is NOT Fraudulent');
```


![png](https://julienbeaulieu.github.io/public/fraud-detection-output/output_66_0.png)


There doesn't seem to be a big different in distributions betweem fraudulent and non fraudulent amounts. 


## 7.4 'Cxx' columns

We saw that some Cxx columns were important. Since they are only numeric, the only useful interpretation we could make is looking at their correlation and their distance from each other using en dendrogram


---
📣**Insights**: 

With hierarchical clustering, a neat feature is being able to visualize the distance between each point. This is called a dendrogram. You can find the code for it below. 



```python
# Get all Cxx columns
c_columns = []
[c_columns.append(txt) for txt in df_train.columns if re.search('^[C]\d+', txt)]
        
print(c_columns)
```

    ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']
    


```python
corr = np.round(scipy.stats.spearmanr(df_train[c_columns]).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(12,7))
dendrogram = hc.dendrogram(z, labels=df_train[c_columns].columns, orientation='left', leaf_font_size=16)
plt.show()
```


![png](https://julienbeaulieu.github.io/public/fraud-detection-output/output_73_0.png)


Let's have a closer look at 'C10', 'C8' and 'C4'. 


```python
g = sns.PairGrid(data=df_train, vars = ['C10', 'C8', 'C4'])
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
```








![png](https://julienbeaulieu.github.io/public/fraud-detection-output/output_75_1.png)


'C10', 'C8' and 'C4' seem to be very close to each other. Maybe we can test removing 1 or 2 of them to limit colinearity (unless we'll only be doing tree based methods which doesn't care about colinearity much).  

## 7.5 P_emaildomain and R_emaildomain


```python
plt.figure(figsize=[17, 5])
base_color = sns.color_palette()[0]
sns.countplot(data=df_train, x='P_emaildomain', order=df_train.P_emaildomain.value_counts().iloc[:10].index, color=base_color);
plt.xlabel('email provider')
plt.title('P_emaildomain distribution - Top 10')
```








![png](https://julienbeaulieu.github.io/public/fraud-detection-output/output_78_1.png)



```python
plt.figure(figsize=[17, 5])
sns.countplot(data=df_train, x='R_emaildomain', order=df_train.P_emaildomain.value_counts().iloc[:10].index, color=base_color);
plt.xlabel('email provider')
plt.title('R_emaildomain distribution - Top 10')
```








![png](https://julienbeaulieu.github.io/public/fraud-detection-output/output_79_1.png)


A lot of these email addresses can be grouped to limit the number of category for these variables (hotmail.com + outlook.com + msn.com = microsoft).  

## 7.6 Device Info


```python
df_train.DeviceInfo.value_counts()[:20]
```




    Windows                        47722
    iOS Device                     19782
    MacOS                          12573
    Trident/7.0                     7440
    rv:11.0                         1901
    rv:57.0                          962
    SM-J700M Build/MMB29K            549
    SM-G610M Build/MMB29K            461
    SM-G531H Build/LMY48B            410
    rv:59.0                          362
    SM-G935F Build/NRD90M            334
    SM-G955U Build/NRD90M            328
    SM-G532M Build/MMB29T            316
    ALE-L23 Build/HuaweiALE-L23      312
    SM-G950U Build/NRD90M            290
    SM-G930V Build/NRD90M            274
    rv:58.0                          269
    rv:52.0                          256
    SAMSUNG                          235
    SM-G950F Build/NRD90M            225
    Name: DeviceInfo, dtype: int64



## 7.7 Id_31 (browser versions)


```python
df_train_og.id_31.value_counts()[:20]
```




    chrome 63.0                   22000
    mobile safari 11.0            13423
    mobile safari generic         11474
    ie 11.0 for desktop            9030
    safari generic                 8195
    chrome 62.0                    7182
    chrome 65.0                    6871
    chrome 64.0                    6711
    chrome 63.0 for android        5806
    chrome generic                 4778
    chrome 66.0                    4264
    edge 16.0                      4188
    chrome 64.0 for android        3473
    chrome 65.0 for android        3336
    firefox 57.0                   3315
    mobile safari 10.0             2779
    chrome 66.0 for android        2349
    chrome 62.0 for android        2097
    edge 15.0                      1600
    chrome generic for android     1158
    Name: id_31, dtype: int64



Both `DeviceInfo` and `Id_31` can be split in order to harness more information.

# 8. Feature Engineering

Now that we're going to build a serious model, let's first have a look at what we're dealing with in terms of null values in our data set. 

## 8.1 Import our data once again to apply feature engineering


```python
# load training_set
df_id_train = pd.read_csv(f'data/train_identity.csv')
df_trans_train = pd.read_csv(f'data/train_transaction.csv')

# load test_set
df_id_test = pd.read_csv(f'data/test_identity.csv')
df_trans_test = pd.read_csv(f'data/test_transaction.csv')

# Merge
df_train = df_trans_train.merge(df_id_train, on='TransactionID', how='left')
df_test = df_trans_test.merge(df_id_test, on='TransactionID', how='left')
```


```python
# Set 'TransactionID' as index
df_train = df_train.set_index('TransactionID')
df_test = df_test.set_index('TransactionID')
```

## 8.2 Identifying and quantifying null values


```python
# Get the number of null values per columns
null_data = df_train.isnull().sum()[df_train.isnull().sum() != 0] 

# Create a DF out of the number of null values 
df_null = pd.DataFrame(null_data, columns = ['number_of_null'])

# Create a percentage column
df_null['percentage_of_null'] = df_null.number_of_null.values / len(df_train) 

# Get columns with over 90% null values
display_all(df_null[df_null['percentage_of_null'] > 0.9])

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number_of_null</th>
      <th>percentage_of_null</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>dist2</td>
      <td>552913</td>
      <td>0.936284</td>
    </tr>
    <tr>
      <td>D7</td>
      <td>551623</td>
      <td>0.934099</td>
    </tr>
    <tr>
      <td>id_07</td>
      <td>585385</td>
      <td>0.991271</td>
    </tr>
    <tr>
      <td>id_08</td>
      <td>585385</td>
      <td>0.991271</td>
    </tr>
    <tr>
      <td>id_18</td>
      <td>545427</td>
      <td>0.923607</td>
    </tr>
    <tr>
      <td>id_21</td>
      <td>585381</td>
      <td>0.991264</td>
    </tr>
    <tr>
      <td>id_22</td>
      <td>585371</td>
      <td>0.991247</td>
    </tr>
    <tr>
      <td>id_23</td>
      <td>585371</td>
      <td>0.991247</td>
    </tr>
    <tr>
      <td>id_24</td>
      <td>585793</td>
      <td>0.991962</td>
    </tr>
    <tr>
      <td>id_25</td>
      <td>585408</td>
      <td>0.991310</td>
    </tr>
    <tr>
      <td>id_26</td>
      <td>585377</td>
      <td>0.991257</td>
    </tr>
    <tr>
      <td>id_27</td>
      <td>585371</td>
      <td>0.991247</td>
    </tr>
  </tbody>
</table>
</div>


This table shows columns with over 90% null values. Let's clean this up a little and remove columns with 90% null values and over.

### Cleaning columns with too many null values and too many repeated values


```python
def get_too_many_null_attr(data):
    many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > 0.9]
    return many_null_cols

def get_too_many_repeated_val(data):
    big_top_value_cols = [col for col in df_train.columns if df_train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    return big_top_value_cols

def get_useless_columns(data):
    too_many_null = get_too_many_null_attr(data)
    print("More than 90% null: " + str(len(too_many_null)))
    too_many_repeated = get_too_many_repeated_val(data)
    print("More than 90% repeated value: " + str(len(too_many_repeated)))
    cols_to_drop = list(set(too_many_null + too_many_repeated))
    #cols_to_drop.remove('isFraud')
    return cols_to_drop
```


```python
cols_to_drop = get_useless_columns(df_train)

```

    More than 90% null: 12
    More than 90% repeated value: 67
    


```python
cols_to_drop = ['id_22', 'id_21', 'V132', 'id_27', 'V124', 'V134', 'V110', 'V135', 'V121', 'V296', 'V297', 'V309', 'id_07',
 'V103', 'V119', 'V318', 'id_25', 'V137', 'id_08', 'V118', 'V129', 'V293', 'V101', 'V136', 'V319', 'D7', 'V109', 'V116',
 'V106', 'C3', 'id_18', 'V298', 'V123', 'V305', 'V107', 'V108', 'V295', 'V311', 'V98', 'V133', 'V320', 'V125', 'V281',
 'V300', 'V102', 'id_23', 'V114', 'V117', 'V284', 'V286', 'V316',  'V105', 'V120', 'V104', 'V290', 'V301', 'dist2', 'id_26',
 'V112', 'V115', 'V321', 'id_24', 'V122', 'V113', 'V299', 'V111']
```


```python
# Drop useless columns
df_train = df_train.drop(cols_to_drop, axis=1)
df_test = df_test.drop(cols_to_drop, axis=1)
```


```python
df_train.shape
```




    (590540, 367)



---
📣 **Insights**

I had tried submitting my results without this step. It turns out I scored 0.001 higher when I _didn't_ do this step. I guess the methodology I apply later on cleans out useless columns without needing to clean beforehand.

I am still doing this step for good measure because I end up with a slightly smaller dataframe which is preferable. 


## 8.3 Extracting value from existing columns

Based on our findings in the EDA phase of the analysis, our dataframe has columns which can be cleaned up and split in order to extract more usefull information. 
> Note: every change we make to `df_train` also has to be made to `df_test`.

We'll first create a function that can:
- Split `DeviceInfo` column on `/` to create 2 new columns containing information about the device name and version. 
- Split `id_30` column on `' '` to create 2 new columns containing information about the OS and the version of the OS.
- Split `id_31` column on `' '` to create 2 new columsn containing information about the browser, and the browser version.
- Group all combinations of phone manufacturer into one common name to avoid having too many distinct phone manufacturer categories
- Map uncommon screensizes to 'Others', instead of having a bunch of screen sizes that didn't make sense.

Once our function is applied, we'll delete the columns that were split to avoid redundancy. 


```python
import gc
def id_split(df):
    df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]
    df['device_version'] = df['DeviceInfo'].str.split('/', expand=True)[1]

    df['OS_id_30'] = df['id_30'].str.split(' ', expand=True)[0]
    df['version_id_30'] = df['id_30'].str.split(' ', expand=True)[1]

    df['browser_id_31'] = df['id_31'].str.split(' ', expand=True)[0]
    df['version_id_31'] = df['id_31'].str.split(' ', expand=True)[1]

    # Group similar names into the same category
    df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'
    
    # Change atypical screensizes with value_counts < 30 to 'Others'
    df.loc[df.id_33.isin(df.id_33.value_counts()[df.id_33.value_counts() < 30].index), 'id_33'] = "Others"
        
    df.loc[df.device_name.isin(df.device_name.value_counts()[df.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    df['had_id'] = 1
    gc.collect()
    
    return df
```


```python
# Apply the function to df_train and df_test
id_split(df_train)
id_split(df_test)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>ProductCD</th>
      <th>card1</th>
      <th>card2</th>
      <th>card3</th>
      <th>card4</th>
      <th>card5</th>
      <th>card6</th>
      <th>addr1</th>
      <th>...</th>
      <th>id_38</th>
      <th>DeviceType</th>
      <th>DeviceInfo</th>
      <th>device_name</th>
      <th>device_version</th>
      <th>OS_id_30</th>
      <th>version_id_30</th>
      <th>browser_id_31</th>
      <th>version_id_31</th>
      <th>had_id</th>
    </tr>
    <tr>
      <th>TransactionID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3663549</td>
      <td>18403224</td>
      <td>31.950</td>
      <td>W</td>
      <td>10409</td>
      <td>111.0</td>
      <td>150.0</td>
      <td>visa</td>
      <td>226.0</td>
      <td>debit</td>
      <td>170.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3663550</td>
      <td>18403263</td>
      <td>49.000</td>
      <td>W</td>
      <td>4272</td>
      <td>111.0</td>
      <td>150.0</td>
      <td>visa</td>
      <td>226.0</td>
      <td>debit</td>
      <td>299.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3663551</td>
      <td>18403310</td>
      <td>171.000</td>
      <td>W</td>
      <td>4476</td>
      <td>574.0</td>
      <td>150.0</td>
      <td>visa</td>
      <td>226.0</td>
      <td>debit</td>
      <td>472.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3663552</td>
      <td>18403310</td>
      <td>284.950</td>
      <td>W</td>
      <td>10989</td>
      <td>360.0</td>
      <td>150.0</td>
      <td>visa</td>
      <td>166.0</td>
      <td>debit</td>
      <td>205.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3663553</td>
      <td>18403317</td>
      <td>67.950</td>
      <td>W</td>
      <td>18018</td>
      <td>452.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>117.0</td>
      <td>debit</td>
      <td>264.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>4170235</td>
      <td>34214279</td>
      <td>94.679</td>
      <td>C</td>
      <td>13832</td>
      <td>375.0</td>
      <td>185.0</td>
      <td>mastercard</td>
      <td>224.0</td>
      <td>debit</td>
      <td>284.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4170236</td>
      <td>34214287</td>
      <td>12.173</td>
      <td>C</td>
      <td>3154</td>
      <td>408.0</td>
      <td>185.0</td>
      <td>mastercard</td>
      <td>224.0</td>
      <td>debit</td>
      <td>NaN</td>
      <td>...</td>
      <td>F</td>
      <td>mobile</td>
      <td>ALE-L23 Build/HuaweiALE-L23</td>
      <td>Huawei</td>
      <td>HuaweiALE-L23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>chrome</td>
      <td>43.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4170237</td>
      <td>34214326</td>
      <td>49.000</td>
      <td>W</td>
      <td>16661</td>
      <td>490.0</td>
      <td>150.0</td>
      <td>visa</td>
      <td>226.0</td>
      <td>debit</td>
      <td>327.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4170238</td>
      <td>34214337</td>
      <td>202.000</td>
      <td>W</td>
      <td>16621</td>
      <td>516.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>224.0</td>
      <td>debit</td>
      <td>177.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4170239</td>
      <td>34214345</td>
      <td>24.346</td>
      <td>C</td>
      <td>5713</td>
      <td>168.0</td>
      <td>144.0</td>
      <td>visa</td>
      <td>147.0</td>
      <td>credit</td>
      <td>NaN</td>
      <td>...</td>
      <td>F</td>
      <td>mobile</td>
      <td>SAMSUNG</td>
      <td>Samsung</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>samsung</td>
      <td>browser</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>506691 rows × 373 columns</p>
</div>




```python
# delete splited cols to remove redundant information 
split_col = ['DeviceInfo', 'id_30', 'id_31']
df_train.drop(split_col, axis=1, inplace=True)
df_test.drop(split_col, axis=1, inplace=True)
```

**More feature engineering based on feature importance and EDA**

- `TransactionAmt` looks like it would be more informative using its log values.
- We'll also create a new column `TransactionAmt_decimal` with only the decimal values of `TransactionAmt`.
- `TransactionDT` is a column related to time, but we have to apply some transformations in order to get the information we want. We'll extract `Transaction_day_of_week` and `Transaction_hour` and create 2 new columns. 


```python
# Add feature: log of transaction amount
df_train['TransactionAmt_Log'] = np.log(df_train['TransactionAmt'])
df_test['TransactionAmt_Log'] = np.log(df_test['TransactionAmt'])

# Add feature: day of the week 
df_train['Transaction_day_of_week'] = np.floor((df_train['TransactionDT'] / (3600 * 24) - 1) % 7)
df_test['Transaction_day_of_week'] = np.floor((df_train['TransactionDT'] / (3600 * 24) - 1) % 7)
                                               
# Add feature: hour of the day 
df_train['Transaction_hour'] = np.floor(df_train['TransactionDT'] / 3600) % 24
df_test['Transaction_hour'] = np.floor(df_test['TransactionDT'] / 3600) % 24

# Add feature: decimal part of the transaction amount
df_train['TransactionAmt_decimal'] = ((df_train['TransactionAmt'] - df_train['TransactionAmt'].astype(int)) * 1000).astype(int)
df_test['TransactionAmt_decimal'] = ((df_test['TransactionAmt'] - df_test['TransactionAmt'].astype(int)) * 1000).astype(int)
```

Valuable information can be extracted from `P_emaildomain` and `R_emaildomain`: 
- Group each variation of emails into different categories. Ex: msn.com, hotmail.es and outlook.com will all be maped to `microsoft`.
- Get the Top Level Domains of all email addresses (.es, .co.jp, .de, etc). 


```python
# Dictionnary of email domains mapped to more general categories
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 
          'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 
           'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 
           'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
           'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other',
            'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 
            'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
          'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 
          'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 
          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo',
          'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft',
          'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other',
          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol',
          'juno.com': 'other', 'icloud.com': 'apple'}
us_emails = ['gmail', 'net', 'edu']

# Perform the map change
for c in ['P_emaildomain', 'R_emaildomain']:
    df_train[c + '_bin'] = df_train[c].map(emails)
    df_test[c + '_bin'] = df_test[c].map(emails)
    
    # Get the TLDs of each email
    df_train[c + '_suffix'] = df_train[c].map(lambda x: str(x).split('.')[-1])
    df_test[c + '_suffix'] = df_test[c].map(lambda x: str(x).split('.')[-1])
    
    # If 
    df_train[c + '_suffix'] = df_train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    df_test[c + '_suffix'] = df_test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
```

Next we'll group the mean transaction per card. 


```python
for card in ['card1', 'card2', 'card3', 'card4', 'card5']: 
    df_train['TransactionAmt_' + card + '_mean'] = df_train['TransactionAmt'] / df_train.groupby([card])['TransactionAmt'].transform('mean')
    df_test['TransactionAmt_' + card + '_mean'] = df_test['TransactionAmt'] / df_test.groupby([card])['TransactionAmt'].transform('mean')
```

Finally, we'll create interactions between important features.


```python
# Create more random interactions between important features 
for feature in ['P_emaildomain_bin__TransactionAmt_Log', 'P_emaildomain_bin__Transaction_hour', 'card1__card2', 
                'card1__card3', 'card1__card5', 'addr1__addr2', 'addr1__card1', 
                'card1__P_emaildomain_bin', 'card1__C1', 'card1__TransactionAmt_Log',
                'V258__card1', 'V258__card2', 'V258__card3', 'V258__P_emaildomain_bin', 'V258__Transaction_hour', 
                 'version_id_31__device_version', 'version_id_31__device_name',
                 'id_33__device_name', 'id_33__device_version', 'id_33__browser_id_31', 
                'card1__card4', 'P_emaildomain_bin__card4', 'TransactionAmt_Log__card4', 
                'id_33__version_id_31']:
    f1, f2 = feature.split('__')
    df_train[feature] = df_train[f1].astype(str) + '_' + df_train[f2].astype(str)
    df_test[feature] = df_test[f1].astype(str) + '_' + df_test[f2].astype(str)
```


```python
# Sanity Check
df_train.shape, df_test.shape
```




    ((590540, 408), (506691, 407))



## 8.4 Apply the cleaning functions so that our new DataFrames are ready for training


```python
train_cats(df_train)
apply_cats(df_test)
```


```python
# Apply proc_df to transform our dataframe, remove NAs and create new columns with NAs
df_train, y, nas = proc_df(df_train, 'isFraud', max_n_cat=8)
```


```python
# Apply proc_df to the test dataframe
df_test, _, nas = proc_df(df_test, na_dict=nas, max_n_cat=8)
```


```python
# Create a copy for safe measures
df_train_copy = df_train.copy()
df_test_copy = df_test.copy()
```

# 9. Reducing Memory Usage and Saving Progress

---
📣 **Insights**

There are two very important ways to be more efficient with our time when working on a large data set. 
1. Save our DataFrame once the cleaning and feature engineering is done so that we don't have to run these steps again when we're continuing working where we left off. **A great library that efficiently stores pandas DataFrame objects on disk is [Feather-format](https://pypi.org/project/feather-format/).** Run this instead of going through all of your data cleaning and feature engineering every time you open your notebook.  


2. Reduce the memory usage of our data frame so that every time we run a model it goes faster. 
 
> Note: Feather format does not support saving columns in Float16. Therefore, we can't save with feather after running the reduce memory function.  


## 9.1 Saving & Loading with Feather 

**Save and import the fully cleaned and feature engineered dataset.**


```python
# Save
os.makedirs('tmp', exist_ok=True)

# Save df_train
df_train.reset_index().to_feather('tmp/ieee_fraud_detection_train') # index must be reset in order to use feather

# Save df_test
df_test.reset_index().to_feather('tmp/ieee_fraud_detection_test')

# Save the target variable normally because it is small 
np.save('tmp/y.npy', y)  
```


```python
# Load
df_train = pd.read_feather('tmp/ieee_fraud_detection_train')
df_test = pd.read_feather('tmp/ieee_fraud_detection_test')
y = np.load('tmp/y.npy')
```


```python
# Set 'TransactionID' as index again
df_train = df_train.set_index('TransactionID')
df_test = df_test.set_index('TransactionID')
```

**Save and load the DataFrame with feature_importance > 80**


```python
# Save
os.makedirs('tmp', exist_ok=True)

# Save df_train
df_train.reset_index().to_feather('tmp/ieee_fraud_detection_train_pruned') # index must be reset in order to use feather

# Save df_test
df_test.reset_index().to_feather('tmp/ieee_fraud_detection_test_pruned')

# Save the target variable normally because it is small 
np.save('tmp/y.npy', y)
```


```python
# Load
df_train = pd.read_feather('tmp/ieee_fraud_detection_train_pruned')
df_test = pd.read_feather('tmp/ieee_fraud_detection_test_pruned')
y = np.load('tmp/y.npy')
```


```python
# Set 'TransactionID' as index again
df_train = df_train.set_index('TransactionID')
df_test = df_test.set_index('TransactionID')
```


```python
df_train.shape, df_test.shape
```




    ((590540, 797), (506691, 796))



## 9.2 Reduce memory usage


```python
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
```


```python
reduce_mem_usage(df_train)
reduce_mem_usage(df_test)
```

    Memory usage of dataframe is 1452.45 MB
    Memory usage after optimization is: 507.99 MB
    Decreased by 65.0%
    Memory usage of dataframe is 1246.22 MB
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:24: RuntimeWarning: invalid value encountered in less
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:26: RuntimeWarning: invalid value encountered in less
    

    Memory usage after optimization is: 446.49 MB
    Decreased by 64.2%
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionDT</th>
      <th>addr1__card1</th>
      <th>card1</th>
      <th>P_emaildomain_bin__TransactionAmt_Log</th>
      <th>TransactionAmt_card1_mean</th>
      <th>card2</th>
      <th>TransactionAmt_card2_mean</th>
      <th>P_emaildomain_bin__Transaction_hour</th>
      <th>card1__card2</th>
      <th>V258__card2</th>
      <th>...</th>
      <th>V76_na</th>
      <th>V15</th>
      <th>V148</th>
      <th>V16</th>
      <th>V254</th>
      <th>M1_T</th>
      <th>V177</th>
      <th>V235</th>
      <th>V198</th>
      <th>V175</th>
    </tr>
    <tr>
      <th>TransactionID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3663549</td>
      <td>18403224</td>
      <td>3894</td>
      <td>10409</td>
      <td>6007</td>
      <td>0.339355</td>
      <td>111.0</td>
      <td>0.209595</td>
      <td>96</td>
      <td>394</td>
      <td>1693</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3663550</td>
      <td>18403263</td>
      <td>19590</td>
      <td>4272</td>
      <td>182</td>
      <td>0.333496</td>
      <td>111.0</td>
      <td>0.321289</td>
      <td>1</td>
      <td>9754</td>
      <td>1693</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3663551</td>
      <td>18403310</td>
      <td>33954</td>
      <td>4476</td>
      <td>21061</td>
      <td>1.485352</td>
      <td>574.0</td>
      <td>0.987305</td>
      <td>120</td>
      <td>9911</td>
      <td>2156</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3663552</td>
      <td>18403310</td>
      <td>10486</td>
      <td>10989</td>
      <td>10924</td>
      <td>2.968750</td>
      <td>360.0</td>
      <td>2.894531</td>
      <td>96</td>
      <td>932</td>
      <td>1942</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3663553</td>
      <td>18403317</td>
      <td>15046</td>
      <td>18018</td>
      <td>7948</td>
      <td>0.567383</td>
      <td>452.0</td>
      <td>0.566406</td>
      <td>96</td>
      <td>7377</td>
      <td>2034</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>4170235</td>
      <td>34214279</td>
      <td>17936</td>
      <td>13832</td>
      <td>8845</td>
      <td>2.771484</td>
      <td>375.0</td>
      <td>2.882812</td>
      <td>112</td>
      <td>3565</td>
      <td>1957</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4170236</td>
      <td>34214287</td>
      <td>38988</td>
      <td>3154</td>
      <td>15903</td>
      <td>0.360107</td>
      <td>408.0</td>
      <td>0.351807</td>
      <td>136</td>
      <td>8816</td>
      <td>304</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4170237</td>
      <td>34214326</td>
      <td>24736</td>
      <td>16661</td>
      <td>18207</td>
      <td>0.424072</td>
      <td>490.0</td>
      <td>0.350586</td>
      <td>136</td>
      <td>6138</td>
      <td>2072</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4170238</td>
      <td>34214337</td>
      <td>0</td>
      <td>16621</td>
      <td>21256</td>
      <td>2.314453</td>
      <td>516.0</td>
      <td>1.890625</td>
      <td>136</td>
      <td>0</td>
      <td>2098</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4170239</td>
      <td>34214345</td>
      <td>39371</td>
      <td>5713</td>
      <td>16729</td>
      <td>0.162964</td>
      <td>168.0</td>
      <td>0.161865</td>
      <td>136</td>
      <td>10930</td>
      <td>71</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>506691 rows × 413 columns</p>
</div>




```python
# sanity check 
df_train.info(verbose=True)
```

# 10. LGBM Hyperparameter Tuning

I tested a few models such as XGBoost, RandomForests and AdaBoost. LGBM was the one that yielded the best results and is the one I  will be demonstrating here.

Let's apply grid search to find the best parameters. 

## 10.1 Grid Search

> Note: My approach was to grid search a few hyperparameters to get a general idea of what works best. I then fine tuned the hyperparameters one by one fitting the model and comparing at the AUC score.  


```python
gridparams = {
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'num_leaves': [20, 25, 31, 35],
    'subsample': [0.7, 1],
    'metric': ['auc'],
    'boosting_type': ['gbdt', 'dart'],
    'objective': ['binary'],
    'colsample_bytree' : [0.64, 0.66, 0.7, 1],
    'reg_alpha': [1, 1.2, 1.4]
}
```


```python
search = GridSearchCV(model, gridparams, verbose=1, n_jobs=-1)
search.fit(X_train, y_train)
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
      warnings.warn(CV_WARNING, FutureWarning)
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    

    Fitting 3 folds for each of 18 candidates, totalling 54 fits
    

    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  5.4min
    [Parallel(n_jobs=-1)]: Done  54 out of  54 | elapsed:  7.6min finished
    




    GridSearchCV(cv='warn', error_score='raise-deprecating',
                 estimator=LGBMClassifier(boosting_type='gbdt', class_weight=None,
                                          colsample_bytree=0.7,
                                          importance_type='split',
                                          learning_rate=0.008, max_depth=-1,
                                          min_child_samples=20,
                                          min_child_weight=0.001,
                                          min_split_gain=0.0, n_estimators=100,
                                          n_jobs=-1, num_leaves=50,
                                          objective='binary', random_state=None,
                                          reg_alpha=0.2, reg_lambda=0.4,
                                          silent=True, subsample=1.0,
                                          subsample_for_bin=200000,
                                          subsample_freq=0),
                 iid='warn', n_jobs=-1,
                 param_grid={'colsample_bytree': [0.7, 0.64],
                             'learning_rate': [0.006, 0.008, 0.01],
                             'num_leaves': [60, 150, 400]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=1)



## 10.2 Using the best model

These are the parameters I am using for the model. Note: they are a combination of hyperparameters taken from a public kernel in Kaggle and some that I have found myself. 


```python
params = {'num_leaves': 600,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.4,
          'bagging_fraction': 0.6,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.005,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47,
          'max_bin': 500
         }
```

Create a sample training and validation dataset to fine tuning parameters and reduce training time.


```python
# Create a training and validation data set. 
n = 50_000
m = 65_000
X_train, X_valid = split_vals_sample(df_train, n, m)
y_train, y_valid = split_vals_sample(y, n, m)

print(F'X_train: {len(X_train)}, X_valid: {len(X_valid)}, y_train: {len(y_train)}, y_valid: {len(y_valid)}')
```

    X_train: 50000, X_valid: 15000, y_train: 50000, y_valid: 15000
    


```python
dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_valid, label=y_valid)

%time clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)
```

    Training until validation scores don't improve for 500 rounds.
    [200]	training's auc: 0.952552	valid_1's auc: 0.906107
    [400]	training's auc: 0.969201	valid_1's auc: 0.916365
    [600]	training's auc: 0.983639	valid_1's auc: 0.92596
    [800]	training's auc: 0.991602	valid_1's auc: 0.932032
    [1000]	training's auc: 0.99581	valid_1's auc: 0.935288
    [1200]	training's auc: 0.997993	valid_1's auc: 0.936999
    [1400]	training's auc: 0.999073	valid_1's auc: 0.937526
    [1600]	training's auc: 0.999591	valid_1's auc: 0.937742
    [1800]	training's auc: 0.999824	valid_1's auc: 0.937923
    [2000]	training's auc: 0.999927	valid_1's auc: 0.93808
    [2200]	training's auc: 0.999971	valid_1's auc: 0.938182
    [2400]	training's auc: 0.999989	valid_1's auc: 0.938316
    [2600]	training's auc: 0.999996	valid_1's auc: 0.938344
    [2800]	training's auc: 0.999999	valid_1's auc: 0.93844
    [3000]	training's auc: 1	valid_1's auc: 0.938445
    [3200]	training's auc: 1	valid_1's auc: 0.938323
    Early stopping, best iteration is:
    [2867]	training's auc: 0.999999	valid_1's auc: 0.938519
    Wall time: 28min 47s
    

# 11. Feature Selection with Feature Importance

## 11.1 Feature importance

---
📣 **Insights**

We can improve the model by removing all the columns that have a feature importance under a specified number. By removing them and training the model again with the same parameters, we end up with a (slightly) better AUC score. 



```python
# Plot top feature importances
lgb.plot_importance(clf, figsize=(15,20), max_num_features=30)
```







![png](https://julienbeaulieu.github.io/public/fraud-detection-output/output_149_1.png)



```python
# Create a function to get feature importance
def feature_importance(df, m):
    fi = pd.DataFrame({'cols': df.columns, 'feature-importances': m.feature_importance(importance_type='split')})\
                        .sort_values(by='feature-importances', ascending=False)
    return fi

fi = feature_importance(df_train, clf)
```

Plot feature_importance distribution


```python
fi.plot('cols', 'feature-importances', figsize=(10,6), legend=False);
```


![png](https://julienbeaulieu.github.io/public/fraud-detection-output/output_152_0.png)


## 11.2 Keep features only with importance > 80


```python
# Keep only the columns with over 80 importance
df_keep = fi[fi['feature-importances'] > 80]
df_keep.cols

df_train = df_train[df_keep.cols]
df_train.shape
```




    (590540, 413)




```python
# Apply the same treatment to our training set
df_test = df_test[df_keep.cols]
df_test.shape
```




    (506691, 413)



Now that we've done some feature selection, we can retrain our model and find what our score is. Again, we have the choice of training on a sample of the data if we want to iterate quickly or train on the full data set.  


```python
# Create a training and validation data set
n = 500_000
X_train, X_valid = split_vals(df_train, n)
y_train, y_valid = split_vals(y, n)

print(F'X_train: {len(X_train)}, X_valid: {len(X_valid)}, y_train: {len(y_train)}, y_valid: {len(y_valid)}')
```

    X_train: 500000, X_valid: 90540, y_train: 500000, y_valid: 90540
    


```python
# Create a sample training and validation data set
n = 50_000
m = 65_000
X_train, X_valid = split_vals2(df_train, n, m)
y_train, y_valid = split_vals2(y, n, m)

print(F'X_train: {len(X_train)}, X_valid: {len(X_valid)}, y_train: {len(y_train)}, y_valid: {len(y_valid)}')
```

    X_train: 50000, X_valid: 15000, y_train: 50000, y_valid: 15000
    


```python
dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_valid, label=y_valid)

clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)
```

    Training until validation scores don't improve for 500 rounds.
    [200]	training's auc: 0.975373	valid_1's auc: 0.903197
    [400]	training's auc: 0.989792	valid_1's auc: 0.910813
    [600]	training's auc: 0.99714	valid_1's auc: 0.917634
    [800]	training's auc: 0.999423	valid_1's auc: 0.922342
    [1000]	training's auc: 0.999903	valid_1's auc: 0.924055
    [1200]	training's auc: 0.999987	valid_1's auc: 0.924778
    [1400]	training's auc: 0.999999	valid_1's auc: 0.925839
    [1600]	training's auc: 1	valid_1's auc: 0.926448
    [1800]	training's auc: 1	valid_1's auc: 0.926727
    [2000]	training's auc: 1	valid_1's auc: 0.926597
    Early stopping, best iteration is:
    [1657]	training's auc: 1	valid_1's auc: 0.92653
    

---
📣 **Insights**

For the real submission to Kaggle I trained the model with the same parameters as above with the importance difference that I didn't include a validation set. The training was therefore done on the entire data. 


# 12. Cross validation with TimeSeriesSplit and StratifiedKFold

The reference for the following code can be found here: https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm

We will try training the model using 2 different kinds of cross validation techniques: Time Series Split and Stratified K Fold. 

The results on the test set didn't end up being better than when I used my initial validation set. 

With more time, I would have tried different numbers of folds for each cross validation technique and submitted my results to Kaggle. I did not end up doing this for this competition however. 

## 12.1 Cross validation 1: TimeSeriesSplit 


```python
# TimeSeriesSplit
NFOLDS = 5
folds = TimeSeriesSplit(n_splits=NFOLDS)
```


```python
# Kfold setup
columns = df_train.columns
splits = folds.split(df_train, y)
y_preds = np.zeros(df_test.shape[0])
y_oof = np.zeros(df_train.shape[0])
score = 0
y = df_train_og.sort_values('TransactionDT')['isFraud']

feature_importances = pd.DataFrame()
feature_importances['feature'] = columns
```


```python
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid = df_train[columns].iloc[train_index], df_train[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    y_preds += clf.predict(df_test) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")
```

    Training until validation scores don't improve for 500 rounds.
    [200]	training's auc: 0.975398	valid_1's auc: 0.888969
    [400]	training's auc: 0.993282	valid_1's auc: 0.898068
    [600]	training's auc: 0.998616	valid_1's auc: 0.902809
    [800]	training's auc: 0.999757	valid_1's auc: 0.903349
    [1000]	training's auc: 0.999962	valid_1's auc: 0.903437
    [1200]	training's auc: 0.999995	valid_1's auc: 0.903001
    Early stopping, best iteration is:
    [869]	training's auc: 0.999865	valid_1's auc: 0.903607
    Fold 1 | AUC: 0.9036067372632375
    Training until validation scores don't improve for 500 rounds.
    [200]	training's auc: 0.974534	valid_1's auc: 0.907594
    [400]	training's auc: 0.991076	valid_1's auc: 0.916753
    [600]	training's auc: 0.99727	valid_1's auc: 0.921575
    [800]	training's auc: 0.999205	valid_1's auc: 0.922767
    [1000]	training's auc: 0.999786	valid_1's auc: 0.922391
    [1200]	training's auc: 0.999947	valid_1's auc: 0.921826
    Early stopping, best iteration is:
    [797]	training's auc: 0.999189	valid_1's auc: 0.92279
    Fold 2 | AUC: 0.9227904767299213
    Training until validation scores don't improve for 500 rounds.
    [200]	training's auc: 0.964211	valid_1's auc: 0.895021
    [400]	training's auc: 0.982684	valid_1's auc: 0.904387
    [600]	training's auc: 0.992685	valid_1's auc: 0.910918
    [800]	training's auc: 0.996848	valid_1's auc: 0.913362
    [1000]	training's auc: 0.998618	valid_1's auc: 0.913995
    [1200]	training's auc: 0.999422	valid_1's auc: 0.914355
    [1400]	training's auc: 0.999771	valid_1's auc: 0.913924
    [1600]	training's auc: 0.999913	valid_1's auc: 0.913364
    Early stopping, best iteration is:
    [1226]	training's auc: 0.999482	valid_1's auc: 0.914416
    Fold 3 | AUC: 0.9144161749993877
    Training until validation scores don't improve for 500 rounds.
    [200]	training's auc: 0.953185	valid_1's auc: 0.906547
    [400]	training's auc: 0.974356	valid_1's auc: 0.919152
    [600]	training's auc: 0.98677	valid_1's auc: 0.928271
    [800]	training's auc: 0.992858	valid_1's auc: 0.932163
    [1000]	training's auc: 0.996053	valid_1's auc: 0.933591
    [1200]	training's auc: 0.997791	valid_1's auc: 0.933822
    [1400]	training's auc: 0.998789	valid_1's auc: 0.933454
    [1600]	training's auc: 0.999314	valid_1's auc: 0.93278
    Early stopping, best iteration is:
    [1190]	training's auc: 0.997726	valid_1's auc: 0.933885
    Fold 4 | AUC: 0.9338850471904426
    Training until validation scores don't improve for 500 rounds.
    [200]	training's auc: 0.948225	valid_1's auc: 0.904561
    [400]	training's auc: 0.969248	valid_1's auc: 0.918444
    [600]	training's auc: 0.982793	valid_1's auc: 0.926972
    [800]	training's auc: 0.989956	valid_1's auc: 0.931474
    [1000]	training's auc: 0.993886	valid_1's auc: 0.933682
    [1200]	training's auc: 0.996172	valid_1's auc: 0.934394
    [1400]	training's auc: 0.997604	valid_1's auc: 0.934668
    [1600]	training's auc: 0.99845	valid_1's auc: 0.93505
    [1800]	training's auc: 0.998968	valid_1's auc: 0.934957
    [2000]	training's auc: 0.999287	valid_1's auc: 0.93491
    Early stopping, best iteration is:
    [1571]	training's auc: 0.998365	valid_1's auc: 0.93514
    Fold 5 | AUC: 0.9351400819846583
    
    Mean AUC = 0.9219677036335295
    Out of folds AUC = 0.8295756814446366
    

## 12.2 Cross validation 2: StratifiedKFold


```python
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS)
```


```python
columns = df_train.columns
splits = folds.split(df_train, y)
y_preds = np.zeros(df_test.shape[0])
y_oof = np.zeros(df_train.shape[0])
score = 0
y = df_train_og.sort_values('TransactionDT')['isFraud']

feature_importances = pd.DataFrame()
feature_importances['feature'] = columns
```


```python
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid = df_train[columns].iloc[train_index], df_train[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    y_preds += clf.predict(df_test) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")
```

    Training until validation scores don't improve for 500 rounds.
    [200]	training's auc: 0.95374	valid_1's auc: 0.882492
    [400]	training's auc: 0.971449	valid_1's auc: 0.893088
    [600]	training's auc: 0.985085	valid_1's auc: 0.900604
    [800]	training's auc: 0.992593	valid_1's auc: 0.906529
    [1000]	training's auc: 0.996345	valid_1's auc: 0.909934
    [1200]	training's auc: 0.99821	valid_1's auc: 0.911887
    [1400]	training's auc: 0.999138	valid_1's auc: 0.912406
    [1600]	training's auc: 0.999578	valid_1's auc: 0.911496
    [1800]	training's auc: 0.999791	valid_1's auc: 0.909049
    Early stopping, best iteration is:
    [1339]	training's auc: 0.998919	valid_1's auc: 0.912508
    Fold 1 | AUC: 0.9125077439911924
    Training until validation scores don't improve for 500 rounds.
    [200]	training's auc: 0.951879	valid_1's auc: 0.865478
    [400]	training's auc: 0.969892	valid_1's auc: 0.86197
    Early stopping, best iteration is:
    [20]	training's auc: 0.931759	valid_1's auc: 0.888944
    Fold 2 | AUC: 0.8889436348654365
    Training until validation scores don't improve for 500 rounds.
    [200]	training's auc: 0.952421	valid_1's auc: 0.895663
    [400]	training's auc: 0.970439	valid_1's auc: 0.902791
    [600]	training's auc: 0.984091	valid_1's auc: 0.903173
    [800]	training's auc: 0.991878	valid_1's auc: 0.902896
    [1000]	training's auc: 0.996008	valid_1's auc: 0.895939
    Early stopping, best iteration is:
    [532]	training's auc: 0.98016	valid_1's auc: 0.904187
    Fold 3 | AUC: 0.9041866732029269
    Training until validation scores don't improve for 500 rounds.
    [200]	training's auc: 0.949968	valid_1's auc: 0.917809
    [400]	training's auc: 0.967401	valid_1's auc: 0.928237
    [600]	training's auc: 0.982163	valid_1's auc: 0.93621
    [800]	training's auc: 0.990997	valid_1's auc: 0.940811
    [1000]	training's auc: 0.995605	valid_1's auc: 0.941614
    [1200]	training's auc: 0.997936	valid_1's auc: 0.94085
    [1400]	training's auc: 0.999057	valid_1's auc: 0.940368
    Early stopping, best iteration is:
    [980]	training's auc: 0.995269	valid_1's auc: 0.941675
    Fold 4 | AUC: 0.941675458403077
    Training until validation scores don't improve for 500 rounds.
    [200]	training's auc: 0.950875	valid_1's auc: 0.895001
    [400]	training's auc: 0.968345	valid_1's auc: 0.905844
    [600]	training's auc: 0.982893	valid_1's auc: 0.915013
    [800]	training's auc: 0.991341	valid_1's auc: 0.920479
    [1000]	training's auc: 0.995739	valid_1's auc: 0.923331
    [1200]	training's auc: 0.997995	valid_1's auc: 0.924709
    [1400]	training's auc: 0.999085	valid_1's auc: 0.925688
    [1600]	training's auc: 0.999582	valid_1's auc: 0.926279
    [1800]	training's auc: 0.999808	valid_1's auc: 0.926536
    [2000]	training's auc: 0.999906	valid_1's auc: 0.926895
    [2200]	training's auc: 0.999954	valid_1's auc: 0.927421
    [2400]	training's auc: 0.999976	valid_1's auc: 0.927776
    [2600]	training's auc: 0.999988	valid_1's auc: 0.928074
    [2800]	training's auc: 0.999994	valid_1's auc: 0.928287
    [3000]	training's auc: 0.999997	valid_1's auc: 0.928349
    [3200]	training's auc: 0.999999	valid_1's auc: 0.928413
    [3400]	training's auc: 1	valid_1's auc: 0.928373
    [3600]	training's auc: 1	valid_1's auc: 0.928224
    [3800]	training's auc: 1	valid_1's auc: 0.928054
    Early stopping, best iteration is:
    [3303]	training's auc: 0.999999	valid_1's auc: 0.928458
    Fold 5 | AUC: 0.9284576809124298
    
    Mean AUC = 0.9151542382750126
    Out of folds AUC = 0.8094206984827366
    

# 13. Ensembling/Stacking models

The reference for the following code and explanations can be found here: https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python.

---
📣 **Insights**

For an excellent article on stacking and ensembling, refer to the de-facto Must read article: [Kaggle Ensembling Guide](https://mlwave.com/kaggle-ensembling-guide/).

In a nutshell stacking uses as a first-level (base), the predictions of a few basic classifiers and then uses another model at the second-level to predict the output from the earlier first-level predictions. Stacking has been responsible for many Kaggle competition wins. 


Here is a very interesting extract of a paper of the creator of stacking: Wolpert (1992) Stacked Generalization:

> It is usually desirable that the level 0 generalizers are of all “types”, and not just simple variations of one another (e.g., we want surface-fitters, Turing-machine builders, statistical extrapolators, etc., etc.). In this way all possible ways of examining the learning set and trying to extrapolate from it are being exploited. This is part of what is meant by saying that the level 0 generalizers should “span the space”.

>[…] stacked generalization is a means of non-linearly combining generalizers to make a new generalizer, to try to optimally integrate what each of the original generalizers has to say about the learning set. The more each generalizer has to say (which isn’t duplicated in what the other generalizer’s have to say), the better the resultant stacked generalization. 



## 13.1 Helper Class

Here we'll invoke the use of Python's classes to help make it more convenient for us. 


```python
# Some useful parameters which will come in handy later on
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(n_splits= NFOLDS, random_state=SEED)
splits = kf.split(df_train) # new

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:,1]
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
```

## 13.2 Out-of-Fold Predictions

As alluded to above stacking uses predictions of base classifiers as input for training to a second-level model. However one cannot simply train the base models on the full training data, generate predictions on the full test set and then output these for the second-level training. This runs the risk of your base model predictions already having "seen" the test set and therefore overfitting when feeding these predictions.


```python
def get_oof(clf, X_train, y_train, X_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(splits):
        x_tr = X_train[train_index]
        y_tr = y_train[train_index]
        x_te = X_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(X_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
```

## 13.3 Generating our base first-level models

Let's prepare 3 learning models as our first level classification: Random Forest, Extra Trees, Gradient Boosting. 


```python
rf_params = {
    'n_jobs': -1,
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':100,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 100,
     'max_features': 0.5,
    'max_depth': 5,
    'min_samples_leaf': 5,
    'verbose': 0
}
```


```python
# Create 3 objects that represent our 3 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
```


```python
# Create Numpy arrays of train, test and target (isFraud) dataframes to feed into our models
y_train = df_train_og['isFraud'].ravel()
X_train = df_train.values # Creates an array of the train data
X_test = df_test.values # Creats an array of the test data
```

## 13.4 Train our classifiers and generate our first level prediction

We now feed the training and test data into our 5 base classifiers and use the Out-of-Fold prediction function we defined earlier to generate our first level predictions.


```python
# Train our 3 classifiers
randf_oof_train, randf_oof_test = get_oof(rf,X_train, y_train, X_test) # Random Forest
et_oof_train, et_oof_test = get_oof(et, X_train, y_train, X_test) # Extra Trees
gb_oof_train, gb_oof_test = get_oof(gb,X_train, y_train, X_test) # Gradient Boost
```

## 13.5 Second-level predictions from the first-level output


We are having as our new columns the first-level predictions from our earlier classifiers and we train the next classifier on this.


```python
base_predictions_train = pd.DataFrame( {'RandomForest': randf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
```


```python
base_predictions_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RandomForest</th>
      <th>ExtraTrees</th>
      <th>GradientBoost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.035713</td>
      <td>0.042941</td>
      <td>0.033789</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.018891</td>
      <td>0.035888</td>
      <td>0.024017</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.017327</td>
      <td>0.020838</td>
      <td>0.018982</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.066818</td>
      <td>0.035453</td>
      <td>0.050022</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.032791</td>
      <td>0.054380</td>
      <td>0.025724</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>590535</td>
      <td>0.017764</td>
      <td>0.037241</td>
      <td>0.015711</td>
    </tr>
    <tr>
      <td>590536</td>
      <td>0.017590</td>
      <td>0.024020</td>
      <td>0.017706</td>
    </tr>
    <tr>
      <td>590537</td>
      <td>0.013963</td>
      <td>0.010343</td>
      <td>0.013736</td>
    </tr>
    <tr>
      <td>590538</td>
      <td>0.051332</td>
      <td>0.053787</td>
      <td>0.046826</td>
    </tr>
    <tr>
      <td>590539</td>
      <td>0.023820</td>
      <td>0.022965</td>
      <td>0.021783</td>
    </tr>
  </tbody>
</table>
<p>590540 rows × 3 columns</p>
</div>




```python
X_train = np.concatenate(( et_oof_train, randf_oof_train, gb_oof_train), axis=1) 
X_test = np.concatenate(( et_oof_test, randf_oof_test, gb_oof_test), axis=1) 
```

Having now concatenated and joined both the first-level train and test predictions as x_train and x_test, we can now fit a second-level learning model.


```python
df_stack1_train = pd.DataFrame(X_train, columns=['ExtraTrees', 'RandomForest', 'GradientBoost'])
df_stack1_test = pd.DataFrame(X_test, columns=['ExtraTrees', 'RandomForest', 'GradientBoost'])
```


```python
df_stack1_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ExtraTrees</th>
      <th>RandomForest</th>
      <th>GradientBoost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.011688</td>
      <td>0.014909</td>
      <td>0.014506</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.020979</td>
      <td>0.017629</td>
      <td>0.018097</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.024601</td>
      <td>0.017673</td>
      <td>0.017714</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.010447</td>
      <td>0.014300</td>
      <td>0.014226</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.012471</td>
      <td>0.015959</td>
      <td>0.014018</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>506686</td>
      <td>0.036896</td>
      <td>0.029405</td>
      <td>0.019036</td>
    </tr>
    <tr>
      <td>506687</td>
      <td>0.063107</td>
      <td>0.040528</td>
      <td>0.023319</td>
    </tr>
    <tr>
      <td>506688</td>
      <td>0.020996</td>
      <td>0.016918</td>
      <td>0.017431</td>
    </tr>
    <tr>
      <td>506689</td>
      <td>0.021273</td>
      <td>0.016918</td>
      <td>0.017431</td>
    </tr>
    <tr>
      <td>506690</td>
      <td>0.097055</td>
      <td>0.037815</td>
      <td>0.026592</td>
    </tr>
  </tbody>
</table>
<p>506691 rows × 3 columns</p>
</div>




```python
# Create validation set using df_stack1_train
n = 500_000
X_train, X_valid = split_vals(df_stack1_train, n)
y_train, y_valid = split_vals(y, n)

print(F'X_train: {len(X_train)}, X_valid: {len(X_valid)}, y_train: {len(y_train)}, y_valid: {len(y_valid)}')
```

    X_train: 500000, X_valid: 90540, y_train: 500000, y_valid: 90540
    


```python
dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_valid, label=y_valid)

clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)
```

    Training until validation scores don't improve for 500 rounds.
    [200]	training's auc: 0.893422	valid_1's auc: 0.856989
    [400]	training's auc: 0.906715	valid_1's auc: 0.856129
    [600]	training's auc: 0.915281	valid_1's auc: 0.854546
    Early stopping, best iteration is:
    [110]	training's auc: 0.884951	valid_1's auc: 0.857647
    

I wasn't able to get a good AUC score using this technique. This is because I had too few base models and that they weren't very good to begin with.  

# 14. Easy ensembling with weighted average

**Easy ensembling**: the most basic and convenient way to ensemble is to ensemble Kaggle submission CSV files. We only need the predictions on the test set for these methods — no need to retrain a model. This makes it a quick way to ensemble already existing model predictions. Here we'll use a simple weighted average of our best submissions. 

> Weighting and averaging prediction files easy, but it’s not the only method that the top Kagglers are using. The serious gains start with stacking and blending. This is not included as it is an advanced technique that I have yet to try out. 


```python
df0 = pd.read_csv('julienbeaulieu_submission5_fullcols.csv')
df1 = pd.read_csv('julienbeaulieu_submission7.csv')
df2 = pd.read_csv('julienbeaulieu_submission8_noTransactionID.csv')
df3 = pd.read_csv('julienbeaulieu_submission11.csv')
df4 = pd.read_csv('julienbeaulieu_submission12.csv')
df5 = pd.read_csv('julienbeaulieu_submission13.csv')
df6 = pd.read_csv('julienbeaulieu_submission17.csv')
df7 = pd.read_csv('julienbeaulieu_submission18.csv')
```


```python
blend1 = df0['isFraud']*0.12 + df1['isFraud']*0.12 + df2['isFraud']*0.12 + df3['isFraud']*0.16 +\
            df4['isFraud']*0.12 + df5['isFraud']*0.12 + df6['isFraud']*0.12 + df7['isFraud']*0.12
```


```python
blend = pd.DataFrame(blend1, columns=['isFraud'])
blend.insert(0, 'TransactionID', df0['TransactionID'])
```


```python
blend
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionID</th>
      <th>isFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3663549</td>
      <td>0.001072</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3663550</td>
      <td>0.001469</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3663551</td>
      <td>0.002976</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3663552</td>
      <td>0.001571</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3663553</td>
      <td>0.001565</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>506686</td>
      <td>4170235</td>
      <td>0.011621</td>
    </tr>
    <tr>
      <td>506687</td>
      <td>4170236</td>
      <td>0.003893</td>
    </tr>
    <tr>
      <td>506688</td>
      <td>4170237</td>
      <td>0.002303</td>
    </tr>
    <tr>
      <td>506689</td>
      <td>4170238</td>
      <td>0.004137</td>
    </tr>
    <tr>
      <td>506690</td>
      <td>4170239</td>
      <td>0.002812</td>
    </tr>
  </tbody>
</table>
<p>506691 rows × 2 columns</p>
</div>




```python
# create submission file
blend.to_csv('julienbeaulieu_submission26.csv', index=False)
```

# 15. Producing the Submission file


```python
ids = df_trans_test.TransactionID.values
prob_test = clf.predict(df_test)
```


```python
submit = pd.DataFrame()
submit['TransactionID'] = ids
submit['isFraud'] = prob_test # y_preds
```


```python
# View the first lines of the submission DataFrame
submit
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionID</th>
      <th>isFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3663549</td>
      <td>0.000064</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3663550</td>
      <td>0.000050</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3663551</td>
      <td>0.000154</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3663552</td>
      <td>0.000090</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3663553</td>
      <td>0.000168</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>506686</td>
      <td>4170235</td>
      <td>0.000705</td>
    </tr>
    <tr>
      <td>506687</td>
      <td>4170236</td>
      <td>0.000222</td>
    </tr>
    <tr>
      <td>506688</td>
      <td>4170237</td>
      <td>0.000192</td>
    </tr>
    <tr>
      <td>506689</td>
      <td>4170238</td>
      <td>0.000227</td>
    </tr>
    <tr>
      <td>506690</td>
      <td>4170239</td>
      <td>0.000162</td>
    </tr>
  </tbody>
</table>
<p>506691 rows × 2 columns</p>
</div>




```python
submit.to_csv('julienbeaulieu_submission24.csv', index=False)
```

# 16. Limitations and going further

This notebook was created thanks for a lot of other notebooks, forum discussion threads and code from the Introduction to Machine Learning for Coders Fastai course. 

While my final score for this competition wasn't great, I am convinced that with more effort I could have improved it.

Here are some ideas to implement to improve the work:

- Adding external sources of data to enrich the current data set.
- Better exploratory data analysis to have a better understanding of the data. My feature engineering would have been better as a result.  
- Better feature selection - I opted to take all the features above 80 feature importance in my LGBM model. I could have tried different feature importance thresholds and different stepwise techniques for choosing features.  
- Ensemble stacking:
    - Adding a much more varied number of algorithms in the first level such as Neural Nets, SVM, KNN, etc. 
    - My base models for XGBoost, Random Forests, and Extra Trees in the first level were not very good. More time could have been spent optimizing them before the stacking procedure. 
- Using techniques to better deal with the imbalanced dataset - over & undersampling. 
- Standardizing the dataset made results worse. I didn't dig deeply as to why and if I was doing it correctly. 
- Cross validation: Trying a different numbers of folds for each cross validation technique. I only tried with 5 folds for both TimeSeriesSplit and StratifiedKFold. 
- Combining crossvalidation, stacking and blending for the ultimate model. 

Thanks for reading!
