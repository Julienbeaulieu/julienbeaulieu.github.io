### [Kaggle competition: IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/overview) 

Can we detect fraud from customer transactions? Lessons from my first competition.

Link to the notebook [on GitHub](https://github.com/Julienbeaulieu/fraud-detection-kaggle-competition).

## Motivation


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

## About this dataset

In this competition we are predicting the probability that an online transaction is fraudulent, as denoted by the binary target `isFraud`. The data comes from [Vesta Corporation's](https://trustvesta.com/) real-world e-commerce transactions and contains a wide range of features from device type to product features.

The data is broken into two files: identity and transaction, which are joined by `TransactionID`.

> Note: Not all transactions have corresponding identity information.



**Evaluation**

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.


## Methodology

### 1. Quick cleaning and modeling

When starting off with a dataset with as many columns as this one (over 400), we first run it through an ensemble learner - LGBM - forgoing any exploratory data analysis and feature engineering at the beginning. Once the model is fit to the data, we have a look at the features which are the most important using LGBM's feature_importances attribute. 
This allows us to concentrate our efforts on only the most important features instead of spending time looking at features with little to no predictive power. 

### 2. Understand the data with EDA

Once we've filtered our columns, we'll look at the ones with the highest importance. The findings in this analysis will guide our feature engineering efforts in the next section. Some questions we'll want to answer:
- How are the top features related to our target variable? 
- What are their distributions like if we plot them with histograms and countplots?  
- What's their relationship with other important features? Do they seem to be related?
- Are there any features that we can split into multiple columns or simplify in any way?
- etc.

### 3. Feature engineering

Once we understand our data, we can start creating new columns by splitting up current ones, transforming them to change their scale or looking at their mean, combining new ones to create interactions, and much more. 

### 4. Train different models, fit them to the training data, and use cross validation

The models I tested were RandomForests, XGBoost, and LightGBM. 

I tried several cross validation techniques such as Stratification and TimeSeriesSplit, neither of which beat my single model LGBM, but it was a great learning experience to code it. 
I also discovered several powerful ensemble techniques which are used by top Kaggle contenders: stacking, blending, averaging our least correlated submissions, etc.  


### 5. Ensembling/Stacking models

For an excellent article on stacking and ensembling, refer to the de-facto Must read article: [Kaggle Ensembling Guide](https://mlwave.com/kaggle-ensembling-guide/). For the code, refer to [this Kaggle guide](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python) or my notebook which is based off of these 2 resources. 

In a nutshell stacking uses as a first-level (base) predictions of a few basic classifiers and then uses another model at the second-level to predict the output from the earlier first-level predictions. Stacking has been responsible for many Kaggle competition wins. 

Here is a very interesting extract of a paper of the creator of stacking: Wolpert (1992) Stacked Generalization:

> It is usually desirable that the level 0 generalizers are of all “types”, and not just simple variations of one another (e.g., we want surface-fitters, Turing-machine builders, statistical extrapolators, etc., etc.). In this way all possible ways of examining the learning set and trying to extrapolate from it are being exploited. This is part of what is meant by saying that the level 0 generalizers should “span the space”.

>[…] stacked generalization is a means of non-linearly combining generalizers to make a new generalizer, to try to optimally integrate what each of the original generalizers has to say about the learning set. The more each generalizer has to say (which isn’t duplicated in what the other generalizer’s have to say), the better the resultant stacked generalization. 

Despite this technique being very powerful, I failed to get a better submission score for myself. This is because my base models were not optimized enough. 

## High-level summary of the scores of my Kaggle competition submissions

- Base features with stock LGBM: 0.87236
- All engineered features with stock : 0.89821 (+0.02585)
- All engineered features with LGBM hyperparameter tuning: 0.91394 (+0.02176)
- Add feature selection: 0.91401 (+0.0007) - **This was my best score. Rank: 2916/6438**
- Use TimeSeriesSplit crossvalidation: 0.89926 (-0.01476 vs top score)
- Use Stratified crossvalidation: 0.90473 (-0.00929 vs top score)
- Stack 3 tree base models for level 1, and use a LGBM for level 2: 0.85597 (-0.05804 vs top score) (base models weren't optimized as much as I wanted, which explains the poor performance)
- Use a weighted average on my top submissions: 0.91145 (-0.00256) - (I think I could have optmized this and gotten a better score)


## Limitations and going further

This notebook was created thanks for a lot of other notebooks, forum discussion threads and code from the Introduction to Machine Learning for Coders Fastai course. 

While my final score for this competition wasn't outstanding, I am convinced that I will do better in my next competition.

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

Link to the code and notebook [on GitHub](https://github.com/Julienbeaulieu/fraud-detection-kaggle-competition).

Thanks for reading!

