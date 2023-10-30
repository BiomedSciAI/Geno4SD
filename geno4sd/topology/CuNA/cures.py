_author__ = "Aritra Bose"
__copyright__ = "Copyright 2023, IBM Research"
__version__ = "0.1"
__maintainer__ = "Aritra Bose"
__email__ = "a.bose@ibm.com"
__status__ = "Development"

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set(color_codes=True)

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import ConfusionMatrixDisplay as CMD
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import statsmodels.api as sm

import warnings
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# warnings.simplefilter('ignore')
import sys, os, time

def split_df(df, pheno):
    x_tr, x_te, y_tr, y_te = train_test_split(df,
                                          pheno, 
                                          test_size=0.3, 
                                          random_state=123)
    x_tr = StandardScaler().fit_transform(x_tr)
    x_te = StandardScaler().fit_transform(x_te)
    
    return x_tr, x_te, y_tr, y_te

def fit_model(x_tr, y_tr):
    
    
    param_grid = [
            {'penalty': ['l1','l2',], 
             'C': np.logspace(-3,3,7),
             'max_iter': [1000],
             'solver': ['liblinear']},
            {'penalty': ['l2'], 
             'C': np.logspace(-3,3,7),
             'max_iter': [500],
             'solver': ['newton-cg']},
             ]
    
    
    log_reg = LR(random_state=123)
    log_reg_cv = GridSearchCV(log_reg, 
                              param_grid=param_grid, 
                              scoring='f1_macro',
                              n_jobs=-1,
                              cv=5)
       

    model = log_reg_cv.fit(x_tr, y_tr)
    best_params = model.best_params_
    
    logreg = LR(C=best_params['C'],
                max_iter = best_params['max_iter'],
                penalty = best_params['penalty'],
                solver = best_params['solver'])
    model = logreg.fit(x_tr, y_tr)
    
    return model, best_params
 
def fit_mc_model(x_tr, y_tr):
    
    
    param_grid = [
            {'penalty': ['l1','l2',], 
             'C': np.logspace(-3,3,7),
             'max_iter': [1000],
             'solver': ['liblinear'],
             'multi_class' : ['ovr']},
        
            {'penalty': ['l2'], 
             'C': np.logspace(-3,3,7),
             'max_iter': [500],
             'solver': ['newton-cg'],
             'multi_class' : ['ovr']}
             ]
    
    
    log_reg = LR(random_state=123)
    log_reg_cv = GridSearchCV(log_reg, 
                              param_grid=param_grid, 
                              scoring='f1_macro',
                              n_jobs=-1,
                              cv=5)
       

    model = log_reg_cv.fit(x_tr, y_tr)
    best_params = model.best_params_
    
    logreg = LR(C=best_params['C'],
                max_iter = best_params['max_iter'],
                penalty = best_params['penalty'],
                solver = best_params['solver'],
                multi_class = best_params['multi_class'])
    
    model = logreg.fit(x_tr, y_tr)
    
    return model, best_params

def accuracy_stats(x_te, 
                   y_te, 
                   y_pred,
                   model,
                   multi_class=False):
    
    res = {}
    res['confusion matrix'] =confusion_matrix(y_te, y_pred)
    res['f1'] = f1_score(y_te, y_pred, average='weighted')
    if multi_class == False:
        res['ROC AUC'] = roc_auc_score(y_te, model.predict_proba(x_te)[:,1])  
    
    return res


def print_scores(model, res, multi_class=False):
        print("-------------------------------------")
        print("Model fitting complete")
        print("-------------------------------------")
        
        if multi_class == False:
            TN, FP, FN, TP = res['confusion matrix'].ravel()
            print('True Positive(TP)  = ', TP)
            print('False Positive(FP) = ', FP)
            print('True Negative(TN)  = ', TN)
            print('False Negative(FN) = ', FN)
            print("-------------------------------------")
        
            print("ROC AUC score of fitted model on test data: ", res['ROC AUC'])
            print("-------------------------------------")
        
        print("F1 score of fitted model on test data: ", res['f1'])
        print("-------------------------------------")
        

def resDF(lRes):
    
    res = pd.DataFrame(np.exp(lRes.params), columns = ['OR'])
    
    #ci_vec = [x for sublist in lRes.conf_int() for x in sublist]
    res['95CI-'] = np.exp(lRes.conf_int().loc[:,0])
    res['95CI+'] = np.exp(lRes.conf_int().loc[:,1])
    res['P-val'] = lRes.pvalues
    
    return '%s(%s):'%(lRes.model.endog_names, ', '.join(lRes.model.exog_names)), res
                                                        
def fit_logit(X,y):
    
    glm = sm.Logit(y, sm.add_constant(X), missing='drop').fit()
  
    return glm
                                                                
def get_cures(df, 
              pheno,
              verbose=0, 
              fit_cures=False,
              multi_class=False,
              get_distance=False):
    
    if 'index' in df.columns:
        df.drop(['index'], axis=1, inplace=True)  
    
    
    x_tr, x_te, y_tr, y_te = split_df(df,pheno)
    if multi_class == False:
        model, best_params = fit_model(x_tr, y_tr)
    else:
        model, best_params = fit_mc_model(x_tr, y_tr)
    
    if verbose == 1:
            print("Best performing logistic regression model on training data: ", best_params)
            
    y_pred = model.predict(x_te)
    
    res = accuracy_stats(x_te, y_te, y_pred, model, multi_class)
    
    if verbose == 1:
        print_scores(model, res, multi_class)
             
    coefs = model.coef_
    if verbose == 1: 
        print("\n**************************************")
        print("Computing CuReS")
        print("**************************************")
    
    cures = np.matmul(x_te, coefs.T)
    cures_res = None
    res_df = None
    
    if fit_cures == True:
        
        if verbose == 1: 
            print("Fitting logistic regression after train-test split")
            
        cures_model, best_params = fit_model(cures, y_te)
        
        if verbose == 1:
            print("Best performing logistic regression model on training data: ", best_params)
            
        cures_y_pred = cures_model.predict(cures)
        cures_res = accuracy_stats(cures, y_te, cures_y_pred, cures_model)

        if verbose == 1:
            print("CuReS prediction statistics")
            print("**************************************")
            print_scores(cures_model, cures_res)

        if verbose == 1: 
            print("\n**************************************")
            print("Associating CuReS with phenotype")
            print("**************************************")
           
        
        logreg = fit_logit(cures, y_te)
        mdl, res_df = resDF(logreg)
        
    if get_distance:
        D = x_tr @ x_tr.T
        D = normalize(D, axis=1, norm='l2')
        return cures, cures_res, res_df, D, y_te

    return cures, cures_res, res_df, y_te

def assoc_cures(df, 
                pheno,
                verbose):
    
    if 'index' in df.columns:
        df.drop(['index'], axis=1, inplace=True)  
        
        
    x_tr, x_te, y_tr, y_te = split_df(df,pheno)

    train_model = fit_logit(x_tr, y_tr)
    coefs_ = train_model.params
    
    cures = np.matmul(x_te, coefs_.T)
    
    logreg = fit_glm(cures, y_te)
    mdl, res_df = resDF(logreg)
    
    return cures, res_df
    
    
    