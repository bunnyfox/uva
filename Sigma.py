#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:34:43 2017

@author: lizhuo
"""
import  os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pylab
from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import random
from sklearn import ensemble, linear_model, metrics
from scipy.stats import norm
import matplotlib.mlab as mlab

os.chdir('/Users/lizhuo/Downloads')
#store=pd.HDFStore('train.h5')
# shape->[1710756,111]
"""Open File"""
with pd.HDFStore("train.h5") as train:
    df=train.get("train")

#mean=df.groupby('id').mean()
d_mean = df.mean(axis=0)
df=df.fillna(d_mean, inplace=True)
    
# check missing value
df.isnull().sum()
# all cols have missing values except for technical_22, and technical_34
labels = []
values = []
for col in df.columns:
    labels.append(col)
    values.append(df[col].isnull().sum())
    print(col, values[-1])

random.seed(99)
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
#train=df[df.timestamp<1400]
#test=df[df.timestamp>=1400]
len(train.id.unique())
len(test.id.unique())
len(train.timestamp.unique()) 
len(test.timestamp.unique()) 
# id are asset id
len(df.id.unique())
# 1424 different assets
len(df.timestamp.unique()) 
# 1813 time period
axis_font = {'fontname':'Arial', 'size':'14'}
print (plt.style.available)
# check available default color scheme
# ['seaborn-pastel', 'seaborn-darkgrid', 'seaborn-whitegrid', 'seaborn-ticks', 'seaborn-poster', 'grayscale', 'seaborn-colorblind', 'seaborn-talk', 'seaborn-bright', 'seaborn-deep', 'seaborn-notebook', 'classic', 'seaborn-dark', 'ggplot', 'seaborn-muted', 'seaborn-paper', 'fivethirtyeight', 'seaborn-dark-palette', 'dark_background', 'bmh', 'seaborn-white']
plt.style.use('ggplot')
plt.figure(figsize=(11,8))
for i, (idVal, dfG) in enumerate(train[['id', 'timestamp', 'y']].groupby('id')):
    #if i> 100: break
    df1 = dfG[['timestamp', 'y']].groupby('timestamp').agg(np.mean).reset_index()
    plt.plot(df1['timestamp'], np.cumsum(df1['y']),label='%d'%idVal)
    plt.xlabel("Time", **axis_font)
    plt.ylabel("Return", **axis_font)
plt.title('Fig 1  Asset Return Over Time')
pylab.savefig('y.png')    


for i, (idVal, dfG) in enumerate(df[['id', 'timestamp', 'y']].groupby('id')):
    if i> 100: break
    #df1 = dfG[['timestamp', 'y']].groupby('timestamp').agg(np.mean).reset_index()
    #plt.plot(df1['timestamp'], np.cumsum(df1['y']),label='%d'%idVal)
    dfG.head()


    
col=[x for x in df.columns if x not in ['id', 'timestamp', 'y']]

# distribution of Y
train["y"].hist(bins = 30, color = "orange")
plt.xlabel("Distribution of Return", **axis_font)
plt.ylabel("Frequency", axis_font)
plt.title('Fig 3 Distribution of Y', axis_font)


#ax = sns.distplot(train['y'], rug=True, rug_kws={"color": "b"}, kde_kws={"color": "k", "lw": 3, "label": "KDE"},hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "b"})

df_f = (train.groupby(pd.cut(train["y"], [-0.087,-0.067,-0.047,-0.027,-0.007,0.013,0.033,0.053,0.073,0.094], right=False))
        .mean())
cor_mat = train.corr(method='pearson', min_periods=1).sort(axis=0, ascending=False, inplace=False)
sns.heatmap(cor_mat, vmax=.8, square=True, cmap='RdBu')

cor_mat_group = df_f.corr(method='pearson', min_periods=1).sort(axis=0, ascending=False, inplace=False)
sns.heatmap(cor_mat_group, vmax=.8, square=True, cmap='Blues')
pylab.savefig('corr.png')
# rank the variables based on their correlation to Y
des_fac=cor_mat.iloc[0,:].sort_values()
des_fac=des_fac.drop('y')
des_fac=des_fac.drop('id')
des_fac=des_fac.drop('timestamp')

des_fac_group=cor_mat_group.iloc[0,:].sort_values()
des_fac_group=cor_mat_group.iloc[0,:].sort_values()
des_fac_group=des_fac_group.drop('y')
des_fac_group=des_fac_group.drop('id')
des_fac_group=des_fac_group.drop('timestamp')
df_fac=df[col]

c_list=['black', 'black','k','k', 'dimgray','dimgray', 'dimgrey', 'dimgrey','gray','gray', \
        'grey', 'grey','darkgray', 'darkgray','darkgrey', 'darkgrey','silver','silver',\
        'lightgray','lightgray', 'lightgrey', 'lightgrey','gainsboro', 'gainsboro','whitesmoke', 'whitesmoke','w','w',\
        'white', 'white','snow','snow', 'rosybrown','rosybrown', 'lightcoral', 'lightcoral',\
        'indianred', 'indianred','brown','brown', 'firebrick','firebrick', 'maroon', 'maroon','darkred','darkred',\
        'maroon','maroon', 'firebrick','firebrick', 'brown',  'brown','indianred','indianred', 'mistyrose', 'mistyrose',\
        'salmon','salmon', 'tomato',  'tomato', 'darksalmon','darksalmon', 'tomato', 'tomato', 'salmon','salmon',\
        'mistyrose','mistyrose','indianred','indianred','brown', 'brown', 'firebrick',  'firebrick','maroon',\
        'darkred','darkred','maroon', 'maroon', 'firebrick',  'firebrick','brown','brown',  'indianred', 'indianred', \
        'lightcoral','rosybrown','snow', 'white', \
        'w','whitesmoke', 'gainsboro', 'lightgrey', 'lightgray', \
        #'gray', 'grey', 'darkgray', 'darkgrey', 'silver',\
        'dimgray', 'k', 'black']
axis_font = {'fontname':'Arial', 'size':'16'}
plt.figure(figsize=(5,35))
plt.subplot(121)
plt.style.use('seaborn-whitegrid')
#uni_cor=train.apply(lambda x: x.corr(df["y"],method='pearson'),axis=0)
#uni_cor=uni_cor.drop('y')
#uni_cor.sort_values(inplace=True)
plt.barh(range(0,len(des_fac)),des_fac,align='center', color=c_list)
plt.yticks(range(0,len(des_fac)),list(des_fac.index), **axis_font)
plt.autoscale()
plt.title('Fig 4 Correlation to Y')
plt.axvline(-0.01, color='maroon')
plt.axvline(0.01, color='maroon')
plt.show()
plt.savefig('correlation_ranking.png')

abs_des=abs(des_fac)
abs_des=abs_des.sort_values()
plt.figure(figsize=(5,35))
plt.subplot(122)
plt.style.use('seaborn-whitegrid')
#uni_cor=train.apply(lambda x: x.corr(df["y"],method='pearson'),axis=0)
#uni_cor=uni_cor.drop('y')
#uni_cor.sort_values(inplace=True)
plt.barh(range(0,len(abs_des)),abs_des,align='center', color=c_list)
plt.yticks(range(0,len(abs_des)),list(abs_des.index))
plt.autoscale()
plt.axvline(0.01, color='maroon')
plt.show()
plt.title('Fig 4 Correlation Ranking')
plt.savefig('correlation_ranking.png')

plt.figure(figsize=(10,35))
plt.style.use('seaborn-whitegrid')
uni_cor=train.apply(lambda x: x.corr(train["y"],method='spearman'),axis=0)
uni_cor=uni_cor.drop('y')
uni_cor.sort_values(inplace=True)
plt.barh(range(0,len(des_fac_group)),des_fac_group,align='center', color=c_list)
plt.yticks(range(0,len(des_fac_group)),list(des_fac_group.index))
plt.autoscale()
plt.show()
plt.savefig('correlation_ranking.png')
#sns.tsplot(train['y'], err_style="boot_traces", n_boot=500)

# correlation between factors

#cor_mat_f=df_fac.corr(method='pearson')
#sns.heatmap(cor_mat_f, vmax=.8, square=True)
#pylab.savefig('corr.png')

y_mean_by_time = train.groupby('timestamp').y.mean()
y_mean_by_time.plot(figsize=(30, 6), color='maroon')
plt.title('Fig 2 Time Series Plot of Target Variable Y', **axis_font)
plt.ylabel('Return Y', **axis_font)
plt.xlabel('Time', **axis_font)




"""
# choose the top and bottom 5 variables
col_k=des_fac.index[0:10,]
col_k+=des_fac.index[-10:,]
list(col_k)
X=train[col_k]
X_scaled = preprocessing.scale(X)
y=train.y
y_scaled = preprocessing.scale(y)
alphas = 10**np.linspace(10,-2,100)*0.5

model=Ridge(fit_intercept=False)
model.fit(X, y)

lasso = Lasso(normalize=True)
lasso.fit(X_scaled, y)
# 10-fold CV
lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)
lassocv.fit(X_scaled, y)
lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_scaled, y)
mean_squared_error(y_test, lasso.predict(X_test))


from sklearn import linear_model, datasets
model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
model_ransac.fit(X_scaled, y)
print(model_ransac.estimator_.coef_)

#train.to_csv('train_s.csv')
#test.to_csv('test_s.csv')


from sklearn.decomposition import PCA
pca = PCA()
#X_s = preprocessing.scale(df_fac)
pca.fit(X_scaled)

var= pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)


cov_mat=np.cov(X_scaled)
eig_vals, eig_vecs=np.linalg.eig(cov_mat)

X=train[col]
X=scale(X)
y=train.y
rf=ensemble.ExtraTreesRegressor()
rf.fit(X, y)
"""

col_k=list(abs_des.index[0:20,])
col_k=['fundamental_10',
 'technical_33',
 'technical_1',
 'fundamental_34',
 'technical_44',
 'fundamental_36',
 'fundamental_26',
 'technical_42',
 'fundamental_33',
 'fundamental_8',
 'fundamental_52',
 'fundamental_24',
 'fundamental_42',
 'technical_31',
 'fundamental_7',
 'fundamental_39',
 'derived_3',
 'fundamental_45',
 'fundamental_20',
 'fundamental_29']
X=train[col_k]
#X.describe()
X=scale(X)
X=pd.DataFrame(X)
X.columns=col_k
X_test=test[col_k]
X_test=scale(X_test)
X_test=pd.DataFrame(X_test)
X_test.columns=col_k
y=test.y
#for i in range(20):
#    plt.hist(X[:,i], bins='auto', color='orange')
#    plt.show()

X.to_hdf('train_x.h5', 'train')
X_test.to_hdf('test_x.h5', 'test')


model = ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, verbose=0)
model.fit(X,train.y)
model.estimators_
model.feature_importances_
model.oob_prediction_
model.decision_path(X)


importances=pd.DataFrame(index=col_k)
# variable importance with Random Forest, default parameters
#est=ensemble.RandomForestRegressor(n_estimators=10000, n_jobs=-1).fit(X, train.y)
#importances['RF']=pd.Series(est.feature_importances_)

# variable importance with Totally Randomized Trees
est=ensemble.ExtraTreesRegressor(max_features=1, max_depth=4, n_estimators=10000, n_jobs=-1).fit(X, train.y)
importances['TRT']=pd.Series(model.feature_importances_, index=col_k)

c_list2=[
        'grey', 'grey','darkgray', 'darkgray','darkgrey', 'darkgrey','silver','silver',\
        'lightgray','lightgray', 'lightgrey', 'lightgrey','gainsboro', 'gainsboro','whitesmoke', 'whitesmoke','w','w',\
        'white', 'white','snow','snow', 'rosybrown','rosybrown', 'lightcoral', 'lightcoral',\
        'indianred', 'indianred','brown','brown', 'firebrick','firebrick', 'maroon', 'maroon','darkred','darkred',\
        'maroon','maroon', 'firebrick','firebrick', 'brown',  'brown','indianred','indianred', 'mistyrose', 'mistyrose',\
        'salmon','salmon', 'tomato',  'tomato', 'darksalmon','darksalmon', 'tomato', 'tomato', 'salmon','salmon',\
        'mistyrose','mistyrose','indianred','indianred','brown', 'brown', 'firebrick',  'firebrick','maroon',\
        'darkred','darkred','maroon', 'maroon', 'firebrick',  'firebrick','brown','brown',  'indianred', 'indianred', \
        'lightcoral','rosybrown','snow', 'white', \
        'w','whitesmoke', 'gainsboro', 'lightgrey', 'lightgray', \
        #'gray', 'grey', 'darkgray', 'darkgrey', 'silver',\
        'dimgray', 'k', 'black']
# variable importance with GBRT
#importances['GBRT']=pd.Series(gbrt.feature_importances_, index=feature_names)

importances.plot(kind='barh', color=c_list2)
plt.title("Fig 5 Feature Importance")


from sklearn.tree import DecisionTreeRegressor
estimator=DecisionTreeRegressor(criterion="mse", max_leaf_nodes=5)

estimator.fit(X, train.y)

y_pred=estimator.predict(X_test)
# mse
from sklearn.metrics import mean_squared_error
score=mean_squared_error(test.y, y_pred )
# 0.00050287543150415376

from sklearn.tree import export_graphviz
export_graphviz(estimator, out_file="tree.x11")



y_pred_rf=model.predict(X_test)

score_rf=mean_squared_error(y_pred_rf, test.y )
# 0.00050283790038591048


S

col_2=['technical_39', 'fundamental_42', 'fundamental_8', 'fundamental_36', 'technical_42']

X2=X[col_2]

import sklearn.cross_validation as scv
from sklearn.learning_curve import validation_curve

param_name='max_features'
param_range=range(1, X.shape[1]+1)
for Forest, color, label in [(ensemble.ExtraTreesRegressor, 'maroon', "ETs")]:
    _, test_scores=validation_curve(
        Forest(n_estimators=100, n_jobs=-1), X, train.y,
        cv=scv.ShuffleSplit(n=len(X), n_iter=10, test_size=0.25),
        param_name=param_name, param_range=param_range,
        scoring="mean_squared_error")
    test_scores_mean=np.mean(-test_scores, axis=1)
    plt.plot(param_range, test_scores_mean, label=label, color=color)

plt.xlabel(param_name)
plt.xlim(1, max(param_range))
plt.ylabel("MSE")
plt.legend(loc='best')
plt.show()
    

test_scores=validation_curve(
    ensemble.ExtraTreesRegressor(n_estimators=100, n_jobs=-1), X, train.y,
    cv=scv.ShuffleSplit(n=len(X), n_iter=10, test_size=0.25),
    param_name=param_name, param_range=param_range,
    scoring="mean_squared_error")
test_scores_mean=np.mean(-test_scores, axis=1)
plt.plot(param_range, test_scores_mean, label="ETs", color='maroon')
plt.xlabel(param_name)
plt.xlim(1, max(param_range))
plt.ylabel("MSE")
plt.legend(loc='best')
plt.show()










