#!/usr/bin/env python
# coding: utf-8

# In[322]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
from correlation_3D import corr_3D, rat_bead_study_data, rat_pa_study_data, mouse_b_enac_study_data, mouse_mps_study_data
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier


# In[324]:


def data_split(df, test_size, val_size):
    np_df = df.values

    bigtrain_set, test_set = train_test_split(np_df, test_size=test_size, random_state=42, stratify=np_df[:,-1])
    train_set, val_set = train_test_split(bigtrain_set, test_size=val_size, random_state=42, stratify=bigtrain_set[:,-1])

    # Get the X and y for train, val and test
    X_train = train_set[:,:-1]
    y_train = train_set[:,-1]
    X_test = test_set[:,:-1]
    y_test = test_set[:,-1]
    X_val = val_set[:,:-1]
    y_val = val_set[:,-1]
    X_bigtrain = bigtrain_set[:,:-1]
    y_bigtrain = bigtrain_set[:,-1]
    
    print(f'Shapes are {[X_train.shape,y_train.shape,X_val.shape,y_val.shape,X_bigtrain.shape,y_bigtrain.shape,X_test.shape,y_test.shape]}')
    
    return X_train,y_train,X_test,y_test,X_val,y_val,X_bigtrain,y_bigtrain


# In[325]:


# Create SVM classifier and optimize the hyperparameters
def svm_hyper_tune(X_train, y_train, X_val, y_val):
    
    preproc_pl = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                        ('std_scaler', StandardScaler())])
    
    for kerneltype in ['rbf','linear','poly']:
        for c_choice in [1, 10, 100]:
            svm_pl = Pipeline([('preproc',preproc_pl),
                               ('svc',SVC(kernel=kerneltype, C=c_choice, random_state=42))])
            svm_pl.fit(X_train,y_train)
            y_pred_svm = svm_pl.predict(X_val)
            acc = accuracy_score(y_val,y_pred_svm)
            print(f'Validation accuracy score = {acc} for kernel {kerneltype} and C={c_choice}')
            


# In[326]:


# Rat bead study (baseline vs post beads)
def svm(X_bigtrain,y_bigtrain,X_test,y_test,name,kernel,c):
    
    preproc_pl = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                        ('std_scaler', StandardScaler())])
    
    svm_pl = Pipeline([('preproc',preproc_pl),
                       ('svc',SVC(kernel=kernel, C=c, random_state=42))])
    
    svm_pl.fit(X_bigtrain,y_bigtrain)

    y_train_pred_svm = svm_pl.predict(X_bigtrain)
    y_test_pred_svm = svm_pl.predict(X_test)

    acc_train = accuracy_score(y_bigtrain,y_train_pred_svm)
    acc_test = accuracy_score(y_test,y_test_pred_svm)
    
    print('\033[1m' + name + '\033[0m')
    print()
    print(f'Training accuracy score = {acc_train}')
    print(f'Testing accuracy score = {acc_test}')
    


# In[327]:


def decision_tree_hyper_tune(X_train, y_train, X_val, y_val):
    
    preproc_pl = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                    ('std_scaler', StandardScaler())])
    
    for criterion in ['gini', 'entropy']:
        for max_depth in [5, 15, 20]:
            dt_pl = Pipeline([('preproc',preproc_pl),
                              ('dt', DecisionTreeClassifier(criterion=criterion,
                                                            max_depth=max_depth, random_state=42))])
            dt_pl.fit(X_train,y_train)
            y_pred_dt = dt_pl.predict(X_val)
            acc = accuracy_score(y_val,y_pred_dt)
            print(f'Validation accuracy score = {acc} for criterion {criterion} and max_depth = {max_depth}')


# In[328]:


def decision_tree(X_bigtrain,y_bigtrain,X_test,y_test,name,criterion,max_depth):
    
    preproc_pl = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                        ('std_scaler', StandardScaler())]) 
    
    dt_pl = Pipeline([('preproc',preproc_pl), ('dt', DecisionTreeClassifier(criterion=criterion,
                                                            max_depth=max_depth, random_state=42))])
    dt_pl.fit(X_bigtrain,y_bigtrain)

    y_train_pred_dt = dt_pl.predict(X_bigtrain)
    y_test_pred_dt = dt_pl.predict(X_test)

    acc_train = accuracy_score(y_bigtrain,y_train_pred_dt)
    acc_test = accuracy_score(y_test,y_test_pred_dt)
    
    print('\033[1m' + name + '\033[0m')
    print()
    print(f'Training accuracy score = {acc_train}')
    print(f'Testing accuracy score = {acc_test}')


# In[329]:


def knn_hyper_tune(X_train, y_train, X_val, y_val):
    
    preproc_pl = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                    ('std_scaler', StandardScaler())])

    for n_neighbors in [3, 5, 7, 9, 11]:
        for weights in ['uniform', 'distance']:
            knn_pl = Pipeline([('preproc',preproc_pl),
                               ('knn', KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights))])
            knn_pl.fit(X_train,y_train)
            y_pred_knn = knn_pl.predict(X_val)
            acc = accuracy_score(y_val,y_pred_knn)
            print(f'Validation recall score = {acc} for n_neighbors = {n_neighbors} and weights {weights}')
            


# In[330]:


def knn(X_bigtrain,y_bigtrain,X_test,y_test,name,n_neighbors,weights):
    
    preproc_pl = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                    ('std_scaler', StandardScaler())])


    knn_pl = Pipeline([('preproc',preproc_pl), ('knn', KNeighborsClassifier(n_neighbors=n_neighbors,
                                                                            weights=weights))])
                                                                    
    knn_pl.fit(X_bigtrain,y_bigtrain)

    y_train_pred_knn = knn_pl.predict(X_bigtrain)
    y_test_pred_knn = knn_pl.predict(X_test)

    acc_train = accuracy_score(y_bigtrain,y_train_pred_knn)
    acc_test = accuracy_score(y_test,y_test_pred_knn)
    
    print('\033[1m' + name + '\033[0m')
    print()
    print(f'Training accuracy score = {acc_train}')
    print(f'Testing accuracy score = {acc_test}')


# In[331]:


def rf_hyper_tune(X_train, y_train, X_val, y_val):
    
    preproc_pl = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                    ('std_scaler', StandardScaler())])

    for criterion in ['gini', 'entropy']:
        for max_depth in [5, 15, 20]:
            rf_pl = Pipeline([('preproc',preproc_pl),
                               ('rf', RandomForestClassifier(n_estimators=100, criterion=criterion,
                                                            max_depth=max_depth, random_state=42))])
            rf_pl.fit(X_train,y_train)
            y_pred_rf = rf_pl.predict(X_val)
            acc = accuracy_score(y_val,y_pred_rf)
            print(f'Validation recall score = {acc} for criterion = {criterion} and max_depth {max_depth}')
            


# In[332]:


def rf(X_bigtrain,y_bigtrain,X_test,y_test,name,criterion,max_depth):
    
    preproc_pl = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                    ('std_scaler', StandardScaler())])


    rf_pl = Pipeline([('preproc',preproc_pl),
                        ('rf', RandomForestClassifier(n_estimators=100, criterion=criterion,
                                                        max_depth=max_depth, random_state=42))])
                                                                    
    rf_pl.fit(X_bigtrain,y_bigtrain)

    y_train_pred_rf = rf_pl.predict(X_bigtrain)
    y_test_pred_rf = rf_pl.predict(X_test)

    acc_train = accuracy_score(y_bigtrain,y_train_pred_rf)
    acc_test = accuracy_score(y_test,y_test_pred_rf)
    
    print('\033[1m' + name + '\033[0m')
    print()
    print(f'Training accuracy score = {acc_train}')
    print(f'Testing accuracy score = {acc_test}')


# In[348]:


def gbc_hyper_tune(X_train, y_train, X_val, y_val):
    
    preproc_pl = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                    ('std_scaler', StandardScaler())])

    for max_depth in [3, 5, 7, 9, 11]:
        for learning_rate in [0.01, 0.1, 1]:      
            gbc_pl = Pipeline([('preproc',preproc_pl),
                               ('gbc', GradientBoostingClassifier(n_estimators=100, max_depth=3, 
                                          learning_rate=learning_rate, random_state=42))])
            gbc_pl.fit(X_train,y_train)
            y_pred_gbc = gbc_pl.predict(X_val)
            acc = accuracy_score(y_val,y_pred_gbc)
            print(f'Validation recall score = {acc} for max_depth = {max_depth} and learning_rate {learning_rate}')


# In[349]:


def gbc(X_bigtrain,y_bigtrain,X_test,y_test,name,max_depth,learning_rate):
    
    preproc_pl = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                    ('std_scaler', StandardScaler())])


    gbc_pl = Pipeline([('preproc',preproc_pl),
                                   ('gbc', GradientBoostingClassifier(n_estimators=100, max_depth=max_depth, 
                                          learning_rate=learning_rate, random_state=42))])
                                                                    
    gbc_pl.fit(X_bigtrain,y_bigtrain)

    y_train_pred_gbc = gbc_pl.predict(X_bigtrain)
    y_test_pred_gbc = gbc_pl.predict(X_test)

    acc_train = accuracy_score(y_bigtrain,y_train_pred_gbc)
    acc_test = accuracy_score(y_test,y_test_pred_gbc)
    
    print('\033[1m' + name + '\033[0m')
    print()
    print(f'Training accuracy score = {acc_train}')
    print(f'Testing accuracy score = {acc_test}')


# In[365]:


def sgd_hyper_tune(X_train, y_train, X_val, y_val):
    preproc_pl = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                    ('std_scaler', StandardScaler())])

    for loss in ['hinge','squared_hinge','perceptron']:
        for penalty in ['l2', 'l1', 'elasticnet']:
            sgd_pl = Pipeline([('preproc',preproc_pl),
                               ('sgd',SGDClassifier(loss=loss, penalty=penalty, random_state=42))])
            sgd_pl.fit(X_train,y_train)
            y_pred_sgd = sgd_pl.predict(X_val)
            acc = accuracy_score(y_val,y_pred_sgd)
            print(f'Validation recall score = {acc} for loss {loss} and penalty {penalty}')


# In[366]:


def sgd(X_bigtrain,y_bigtrain,X_test,y_test,name,loss,penalty):
    
    preproc_pl = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                    ('std_scaler', StandardScaler())])
    
    sgd_pl = Pipeline([('preproc',preproc_pl),
                        ('sgd',SGDClassifier(loss=loss, penalty=penalty, random_state=42))])
    sgd_pl.fit(X_bigtrain,y_bigtrain)
    
    y_train_pred_sgd = sgd_pl.predict(X_bigtrain)
    y_test_pred_sgd = sgd_pl.predict(X_test)

    acc_train = accuracy_score(y_bigtrain,y_train_pred_sgd)
    acc_test = accuracy_score(y_test,y_test_pred_sgd)
    
    print('\033[1m' + name + '\033[0m')
    print()
    print(f'Training accuracy score = {acc_train}')
    print(f'Testing accuracy score = {acc_test}')
    


