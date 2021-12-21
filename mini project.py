# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:53:38 2021

@author: aishg
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV,ElasticNetCV
from sklearn.model_selection import RepeatedKFold
import sklearn

df = pd.read_csv("C:/Users/aishg/OneDrive/Desktop/Big data/assignment-1/feature spaces/PermutedMC.csv")


y = df["64"].astype('float64')

X = df.drop(['64'], axis=1).astype('float64')

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.25, 
                                                    random_state=42)

enet_model = ElasticNet().fit(X_train, y_train)

enet_model.coef_

enet_model.intercept_

enet_model.predict(X_train)[:10]

enet_model.predict(X_test)[:10]

y_pred = enet_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))

r2_score(y_test,y_pred)

enet_cv_model = ElasticNetCV(cv = 10).fit(X_train,y_train)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

scores = cross_val_score(enet_cv_model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)


enet_cv_model.alpha_

enet_cv_model.intercept_

enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_).fit(X_train,y_train)

y_pred = enet_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))


ax = plt.gca()
ax.plot(y_pred)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('lambda')
plt.ylabel('weights')

X_train=np.array(X_train)
y_train=np.array(y_train)
y_train=y_train.reshape(-1,1)
y_test=np.array(y_test)
y_test=y_test.reshape(-1,1)

def getParametersElasticNet(X,y):
    
    bestMSE=10e100
    
    regList=[l*0.1 for l in range(1,500)]
    ratio=[i*0.1 for i in range(1,200)]
    global bestAlpha
    global bestRatio
    global bestElasticWeights
    
    for l1 in regList:
        for r in ratio:
              elasticModel=sklearn.linear_model.ElasticNet(
              alpha=l1,l1_ratio=r,fit_intercept=False,
              max_iter=3000,tol=1e-5)
            
              elasticModel.fit(X,y)
              getPred=elasticModel.predict(X_test).reshape(-1,1)
        
              MSE=sum((y_test-getPred)**2)
              if MSE< bestMSE:
                bestMSE=MSE
                bestAlpha=l1
                bestRatio=r
                bestElasticWeights=elasticModel.coef_
                
    return(bestElasticWeights)

  


# plug in ideal parameters
fitElastic=sklearn.linear_model.ElasticNet(alpha=bestAlpha,l1_ratio=bestRatio,fit_intercept=False)
fitElastic.fit(X,y)

Welastic=fitElastic.coef_

pz=fitElastic.predict(X).reshape(-1,1)


ax = plt.gca()
ax.plot(pz)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('lambda')
plt.ylabel('weights')