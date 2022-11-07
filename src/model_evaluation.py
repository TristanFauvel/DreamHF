import pandas as pd
import numpy as np
import xgboost as xgb
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

from HosmerLemeshowSurvival import HosmerLemeshowSurvival
from xgboost_wrapper import XGBSurvival

def evaluate_model(model, X_train, X_test, y_train, y_test):
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    
    event_field, time_field = y_train.dtype.names
     
    y = np.concatenate([y_train[time_field], y_test[time_field]])
    times = np.percentile(
        y, np.linspace(5, 95, 15)
    )  
    # We set the upper bound to the 95% percentile of observed time points,
    # because the censoring rate is quite large at 91.5%.

    tau = times[
        -1
    ] 
    #  Truncation time. The survival function for the underlying
    # censoring time distribution needs to be positive at tau

     

    #%% Harrel's concordance index
    # Harrel's concordance index C is defined as the proportion of
    # observations that the model can order correctly in terms of survival times.
    concordance_index_censored_train = concordance_index_censored(
        y_train[event_field], y_train[time_field], preds_train
    )

    concordance_index_censored_test = concordance_index_censored(
        y_test[event_field], y_test[time_field], preds_test
    )

    #%% Uno's concordance index (based on inverse probability of censoring weights)

    concordance_index_ipcw_train = concordance_index_ipcw(
        y_train, y_train, preds_train, tau=tau
    )
    concordance_index_ipcw_test = concordance_index_ipcw(
        y_train, y_test, preds_test, tau=tau
    )

    if not isinstance(model, XGBSurvival):
        #%% Integrated Brier score
        try:
            survs = model.predict_survival_function(X_train)
            preds = np.asarray([[fn(t) for t in times] for fn in survs])
            integrated_brier_score_train = integrated_brier_score(
                y_train, y_train, preds, times
            )
        finally:
            integrated_brier_score_train = np.nan

        try:
            survs = model.predict_survival_function(X_test)
            preds = np.asarray([[fn(t) for t in times] for fn in survs])
            integrated_brier_score_test = integrated_brier_score(
                y_train, y_test, preds, times
            )
        finally:
            integrated_brier_score_test = np.nan

        HL_train = HosmerLemeshowSurvival(10, model, X_train, y_train, df=2, Q=10)
        HL_test = HosmerLemeshowSurvival(10, model, X_test, y_test, df=2, Q=10)

        result = {
            "Harrell C": [
                concordance_index_censored_train[0],
                concordance_index_censored_test[0],
            ],
            "Concordance index IPCW": [
                concordance_index_ipcw_train[0],
                concordance_index_ipcw_test[0],
            ],
            "Integrated Brier Score": [
                integrated_brier_score_train,
                integrated_brier_score_test,
            ],
            "Hosmer-Lemeshow": [f"{HL_train['pvalue']:.2e}", f"{HL_test['pvalue']:.2e}"],
        }
    else:
        result = {
        "Harrell C": [
            concordance_index_censored_train[0],
            concordance_index_censored_test[0],
        ],
        "Concordance index IPCW": [
            concordance_index_ipcw_train[0],
            concordance_index_ipcw_test[0],
        ],
    }

    return pd.DataFrame(result, index=["train", "test"])

