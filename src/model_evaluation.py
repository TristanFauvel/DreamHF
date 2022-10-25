import pandas as pd
import numpy as np

from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.column import encode_categorical
from sksurv.ensemble import RandomSurvivalForest
from dotenv import load_dotenv
import os

from HosmerLemeshowSurvival import HosmerLemeshowSurvival

def evaluate_model(model, X_train, X_test, y_train, y_test):
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    
    y = np.concatenate([y_train.Event_time, y_test.Event_time])
    times = np.percentile(y, np.linspace(5, 95, 15)) #We set the upper bound to the 95% percentile of observed time points, because the censoring rate is quite large at 91.5%.

    tau = times[-1] #  Truncation time. The survival function for the underlying censoring time distribution needs to be positive at tau

    preds_train = model.predict(X_train)  #Risk score prediction
    preds_test = model.predict(X_test)

    #%% Harrel's concordance index
    #Harrel's concordance index C is defined as the proportion of observations that the model can order correctly in terms of survival times. 
    concordance_index_censored_train = concordance_index_censored(y_train["Event"], y_train["Event_time"], preds_train)

    concordance_index_censored_test = concordance_index_censored(y_test["Event"], y_test["Event_time"], preds_test)

    #%% Uno's concordance index (based on inverse probability of censoring weights)
    
    concordance_index_ipcw_train = concordance_index_ipcw(y_train, y_train, preds_train, tau = tau)
    concordance_index_ipcw_test = concordance_index_ipcw(y_train, y_test, preds_test, tau = tau)

    #%% Integrated Brier score
    try : 
        survs = model.predict_survival_function(X_train)
        preds = np.asarray([[fn(t) for t in times] for fn in survs]) 
        integrated_brier_score_train = integrated_brier_score(y_train, y_train, preds, times)
    except :
        integrated_brier_score_train = np.nan
    
    try:
        survs = model.predict_survival_function(X_test)
        preds = np.asarray([[fn(t) for t in times] for fn in survs]) 
        integrated_brier_score_test = integrated_brier_score(y_train, y_test, preds, times)
    except:
        integrated_brier_score_test = np.nan
 
 
    HL_train = HosmerLemeshowSurvival(10, model, X_train, y_train, df = 2, Q = 10)
    HL_test = HosmerLemeshowSurvival(10, model, X_test, y_test, df = 2, Q = 10)
    

    #result = {'Harrell_C_train':  concordance_index_censored_train[0], 'Harrell_C_test' : concordance_index_censored_test[0]}
    
    #result_test = {'Harrell C': concordance_index_censored_test[0],
             # 'Concordance index IPCW': concordance_index_ipcw_test[0],
              #'Cumulative Dynamic AUC' : [cumulative_dynamic_auc_train, cumulative_dynamic_auc_test], 
             # 'Integrated Brier Score': integrated_brier_score_test}
 
    result = {'Harrell C': [concordance_index_censored_train[0], concordance_index_censored_test[0]], 'Concordance index IPCW': [concordance_index_ipcw_train[0], concordance_index_ipcw_test[0]],
              'Integrated Brier Score': [integrated_brier_score_train, integrated_brier_score_test], 'Hosmer-Lemeshow': [f"{HL_train['pvalue']:.2e}", f"{HL_test['pvalue']:.2e}"]}
    
    return pd.DataFrame(result, index = ['train', 'test'])