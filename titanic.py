# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 18:26:12 2017

@author: vamsi.mudimela
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import statistics

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib import rcParams
rcParams['figure.figsize']=12,4

#Function to create GBM models and run cross-validation
def gbmfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True,cv_folds = 5):
    #fit algorithm to data
    alg.fit(dtrain[predictors], dtrain['Survived'])
    #predict on the training set
    d_train_pred = alg.predict(dtrain[predictors])
    d_train_prob = alg.predict_proba(dtrain[predictors])[:,1]
    #perform cross-validation
    if performCV:
        cv_score=cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Survived'],
                                                  cv = cv_folds,
                                                  scoring='roc_auc')
    #print model report
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Survived'].values, d_train_pred)
    print "AUC score (Train) : %f" % metrics.roc_auc_score(dtrain['Survived'],d_train_prob)
    
    #print cross validation summary    
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score), np.min(cv_score), np.max(cv_score))
    #print feature importance
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar',title = "feature importance")
        plt.ylabel("feature importance score")
 
#read training data
d_train = pd.read_csv('C:/Users/vamsi.mudimela/Documents/Library/learning/Python/kaggle_getting_started/titanic/train.csv')
d_test = pd.read_csv('C:/Users/vamsi.mudimela/Documents/Library/learning/Python/kaggle_getting_started/titanic/test.csv')

#--------------data exploration--------#

#1. check dimensions
d_train.shape
d_train.dtypes

d_train['Pclass'] = d_train['Pclass'].astype(str)


#2. check data types
d_train.dtypes

#3. check missing values
d_train.columns.values.tolist()

for col in d_train.columns :
    print(col, pd.isnull(d_train[col]).values.ravel().sum())
    
d_train['Cabin'].value_counts() #--ignore Cabin variables. too many missing values
d_train = d_train.drop('Cabin',1)
d_train['Embarked'].value_counts()

#impute missing values
d_train['Age']=d_train['Age'].fillna(d_train['Age'].mean()) #--impute mean
#d_train['Embarked']=d_train['Embarked'].fillna('S') #--impute mode
d_train['Embarked']=d_train['Embarked'].fillna(statistics.mode(d_train['Embarked'])) #--impute mode

#4. check distributions
d_train.columns.values.tolist()
pd.crosstab(d_train['Survived'], d_train['Pclass'], margins=True)

for col in d_train.columns:
    if d_train[col].dtypes == 'object':
        print d_train.groupby(['Survived',col])[col].count()
    else:
        print d_train.groupby('Survived')[col].mean()

#get all categorical variables
char_col = d_train.loc[:, d_train.dtypes==np.object].columns.values.tolist()
#get all numeric variables
num_col = d_train.loc[:, d_train.dtypes!=np.object].columns.values.tolist()
char_col
num_col

#get summary statistics for variables
#d_train.groupby('Survived')[char_col].count()
d_train.groupby('Survived')[num_col].mean()

# 5. create dummy variables
dummy_class = pd.get_dummies(d_train['Pclass'], prefix = "pclass")
dummy_sex = pd.get_dummies(d_train['Sex'], prefix = "sex")
dummy_embarked = pd.get_dummies(d_train['Embarked'], prefix = "embarked")

all_cols = d_train.columns.values.tolist()
all_cols
all_cols.remove('Pclass')
all_cols.remove('Sex')
all_cols.remove('Embarked')
all_cols
d_train = d_train[all_cols].join(dummy_class)
d_train = d_train.join(dummy_sex)
d_train = d_train.join(dummy_embarked)
d_train.columns.values.tolist()

d_train = d_train.drop('Name',1)
d_train = d_train.drop('Ticket',1)

d_train.head(5)

#------------Variable selection--------------#

#----1. GBM-------------------------#

# Step 1 - setup a baseline model
d_train.columns.values.tolist()
preds = [x for x in d_train.columns if x not in (['PassengerId','Survived'])]
gbm0 = GradientBoostingClassifier(random_state = 10)
gbmfit(gbm0, d_train, preds)

# Step 2 - Parameter tuning

# a. n_estimators : no. of trees
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator=(GradientBoostingClassifier(learning_rate=0.1,
                                                              min_samples_split=50,
                                                              min_samples_leaf=20,
                                                              max_depth=5,
                                                              max_features='sqrt',
                                                              subsample=0.8,
                                                              random_state=10)), 
                        param_grid=param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv = 5)
gsearch1.fit(d_train[preds],d_train['Survived'])
gsearch1.best_params_

# b. max_depth and min_samples_split
param_test2 = {'max_depth':range(3,7,1), 'min_samples_split':range(5,51,1)}
gsearch2 = GridSearchCV(estimator=(GradientBoostingClassifier(learning_rate=0.1,
                                                              n_estimators = 70,
                                                              min_samples_leaf=5,
                                                              max_features='sqrt',
                                                              subsample=0.8,
                                                              random_state=10)),
                        param_grid=param_test2,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch2.fit(d_train[preds],d_train['Survived'])
gsearch2.best_params_

# c. min_samples_leaf
param_test3 = {'min_samples_leaf':range(3,21,1)}
gsearch3 = GridSearchCV(estimator=(GradientBoostingClassifier(learning_rate=0.1,
                                                              n_estimators =70,
                                                              max_depth=5,
                                                              min_samples_split=16,
                                                              max_features='sqrt',
                                                              subsample=0.8,
                                                              random_state=10)),
                        param_grid=param_test3,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch3.fit(d_train[preds],d_train['Survived'])
gsearch3.best_params_

# d. max_depth
param_test4 = {'max_depth':range(3,9,1)}
gsearch4 = GridSearchCV(estimator=(GradientBoostingClassifier(learning_rate=0.1,
                                                              n_estimators =70,
                                                              min_samples_leaf=5,
                                                              min_samples_split=16,
                                                              max_features='sqrt',
                                                              subsample=0.8,
                                                              random_state=10)),
                        param_grid=param_test4,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch4.fit(d_train[preds],d_train['Survived'])
gsearch4.best_params_


# e. sub_sample
param_test5 = {'subsample':[0.6,0.65,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator=(GradientBoostingClassifier(learning_rate=0.1,
                                                              n_estimators =70,
                                                              min_samples_leaf=5,
                                                              min_samples_split=16,
                                                              max_features='sqrt',
                                                              max_depth=5,
                                                              random_state=10)),
                        param_grid=param_test5,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch5.fit(d_train[preds],d_train['Survived'])
gsearch5.best_params_

# Run model using tuned parameters
gbmfit(gsearch5.best_estimator_,d_train, preds)

#----------Variable selection -2 --------------------------

