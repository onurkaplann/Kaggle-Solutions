# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:19:47 2020

@author: Onur
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
import csv
from sklearn.impute import SimpleImputer 
import matplotlib.pyplot as plt


# item_categories = pd.read_csv('item_categories.csv')
# items = pd.read_csv('items.csv')
# shops = pd.read_csv('shops.csv')

data_test = pd.read_csv('test.csv')
sales_train = pd.read_csv('sales_train.csv',parse_dates = ['date'])
sample_submission = pd.read_csv('sample_submission.csv')

plt.figure(figsize=(8,8))
plt.scatter(sales_train.item_cnt_day,sales_train.item_price)
plt.show()

sales_train = sales_train[sales_train.item_price<45000]
sales_train = sales_train[sales_train.item_cnt_day<600]
#sales_train = sales_train[sales_train.item_cnt_day>=0]

columns = ['date', 'date_block_num', 'shop_id', 'item_id','item_price','item_cnt_day']
sales_train.drop_duplicates(columns,keep='first', inplace=True) 

plt.figure(figsize=(8,8))
plt.scatter(sales_train.item_cnt_day,sales_train.item_price)
plt.show()

monthly = sales_train.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].agg('sum').reset_index()
monthly.columns = ['date_block_num','shop_id','item_id','item_cnt_month']

price_monthly = sales_train.groupby(["date_block_num","shop_id","item_id"])["item_price"].agg('mean').reset_index()

combine = pd.merge(monthly, price_monthly)

sales_train['year'] = sales_train['date'].dt.year
sales_train['day_of_year'] = sales_train['date'].dt.dayofyear
sales_train['weekday'] = sales_train['date'].dt.weekday
sales_train['week_of_year'] = sales_train['date'].dt.week
sales_train['day_of_month'] = sales_train['date'].dt.day
sales_train['quarter'] = sales_train['date'].dt.quarter
sales_train['month'] = sales_train['date'].dt.month
sales_train.drop('date', axis=1, inplace=True)

datatest_monthly = price_monthly.groupby(["shop_id","item_id"])["item_price"].agg('mean').reset_index()
data_test['date_block_num'] = 34
data_test_end = pd.merge(data_test, datatest_monthly,how = 'left')
data_test_end = data_test_end.drop(['ID'], axis = 1)

numeric = SimpleImputer(missing_values=np.nan, strategy='mean')
numeric = numeric.fit(data_test_end)
data_test_end = numeric.transform(data_test_end)
data_test_end = pd.DataFrame(data_test_end,columns=["shop_id","item_id","date_block_num","item_price"])


X = combine.drop('item_cnt_month', axis=1)
Y = combine.item_cnt_month

def XGBoost():
    
    train, test, y_train, y_test = train_test_split(combine.drop('item_cnt_month', axis=1), combine.item_cnt_month, test_size=0.2, random_state=10)  

    model = xgb.XGBRegressor(
        max_depth=8,
        n_estimators=1000,
        min_child_weight=300, 
        colsample_bytree=0.8, 
        subsample=0.8, 
        eta=0.3,    
        seed=42)
    
    model.fit(
        train, 
        y_train, 
        eval_metric="rmse", 
        eval_set=[(train, y_train), (test, y_test)], 
        verbose=True, 
        early_stopping_rounds = 10)
    
    y_pred = model.predict(test)
    y_pred = y_pred.tolist()
    
    print('R2 XGBoost : ',r2_score(y_test,y_pred))
    
    fig, ax = plt.subplots(figsize=(10,10))
    xgb.plot_importance(model, importance_type='gain',ax=ax)
    plt.show()

def CatBoost():
    
    train, test, y_train, y_test = train_test_split(combine.drop('item_cnt_month', axis=1), combine.item_cnt_month, test_size=0.2, random_state=10)  

    model =  cb.CatBoostRegressor(iterations=1000,
                                 learning_rate=0.01,
                                 depth=16,
                                 eval_metric='RMSE',
                                 random_seed = 42,
                                 bagging_temperature = 0.2,
                                 od_type='Iter',
                                 metric_period = 75,
                                 od_wait=100)
    
    model.fit(train, y_train,eval_set=(test, y_test),plot=True)
    
    y_pred = model.predict(test)
    y_pred = y_pred.tolist()
    
    print('R2 CatBoost  : ',r2_score(y_test,y_pred))
    
    fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': combine.drop('item_cnt_month', axis=1).columns})
    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
    fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)
    plt.title('CatBoost - Feature Importance')
    plt.ylabel('Features')
    plt.xlabel('Importance');

def LightGBM():
    
    train, test, y_train, y_test = train_test_split(combine.drop('item_cnt_month', axis=1), combine.item_cnt_month, test_size=0.2, random_state=10)  
     
    hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
#    'metric': ['l2', 'auc'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,  
    "max_bin": 512,
    "num_iterations": 1000,
    "n_estimators": 1000
    }

    model = lgb.LGBMRegressor(**hyper_params)
    
    model.fit(train, y_train,
            eval_set=[(test, y_test)],
            eval_metric='l1',
            early_stopping_rounds=1000)
    
    y_pred = model.predict(test)
    
    print('R2 LightGBM : ',r2_score(y_test,y_pred))
    #print('RMSE : 'metrics.mean_squared_error(y_test, y_pred, squared=False))
    
    
    lgb.plot_importance(model, importance_type='gain', max_num_features=20,figsize=(15,15))
    plt.show()


def RandomForestRegressor(sample_submission):
    
    train, test, y_train, y_test = train_test_split(combine.drop('item_cnt_month', axis=1), combine.item_cnt_month, test_size=0.2, random_state=10)  
    
    from sklearn.ensemble import RandomForestRegressor
    rf_reg = RandomForestRegressor(n_estimators=100,random_state=0)
    rf_reg.fit(train,y_train)
    
    y_pred = rf_reg.predict(test)
    print('R2 : ',r2_score(y_test,y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    
    y_pred = rf_reg.predict(data_test_end)
    y_pred = pd.DataFrame(y_pred,columns=["item_cnt_month"])
    y_pred = y_pred.clip(0,20)
    # print('R2 : ',r2_score(y_test,y_pred))
    # print('Validation rmse:', np.sqrt(mean_squared_error(y_test, y_pred)))
    # print(accuracy_score(y_pred, y_test)*100)
    
    sample_submission = sample_submission.drop(['item_cnt_month'], axis = 1)
    sample_submission=pd.concat([sample_submission,y_pred],axis=1)
    sample_submission.to_csv('submission.csv', index=False)


XGBoost()
CatBoost()
LightGBM()
RandomForestRegressor(sample_submission)








