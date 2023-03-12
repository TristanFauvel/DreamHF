import numpy as np
import pandas as pd
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    integrated_brier_score,
)

from HosmerLemeshowSurvival import HosmerLemeshowSurvival


def evaluate_model(model: object, x_train: pd.DataFrame, x_test: pd.DataFrame,
                   y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Evaluates the given model with the given data and returns the results.

    Parameters:
    model (object): A trained model to be evaluated.
    x_train (pd.DataFrame): A dataframe containing the training features.
    x_test (pd.DataFrame): A dataframe containing the testing features.
    y_train (np.ndarray): A numpy array containing the training survival data.
    y_test (np.ndarray): A numpy array containing the testing survival data.

    Returns:
    pd.DataFrame: A dataframe containing the evaluation results.

    """

    preds_train = model.predict(x_train)
    preds_test = model.predict(x_test)

    event_field, time_field = y_train.dtype.names

    y_data= np.concatenate([y_train[time_field], y_test[time_field]])
    times = np.percentile(y_data, np.linspace(5, 95, 15))

    tau = times[-1]

    # Harrel's concordance index
    concordance_index_censored_train = concordance_index_censored(
        y_train[event_field], y_train[time_field], preds_train)

    concordance_index_censored_test = concordance_index_censored(
        y_test[event_field], y_test[time_field], preds_test)

    # Uno's concordance index (based on inverse probability of censoring weights)
    concordance_index_ipcw_train = concordance_index_ipcw(
        y_train, y_train, preds_train, tau=tau)
    concordance_index_ipcw_test = concordance_index_ipcw(
        y_train, y_test, preds_test, tau=tau)

    integrated_brier_score_train = np.nan
    integrated_brier_score_test = np.nan

    if hasattr(model, 'predict_survival_function'):
        # %% Integrated Brier score

        try:
            survs = model.predict_survival_function(x_train)
            preds = np.asarray([[fn(t) for t in times] for fn in survs])
            integrated_brier_score_train = integrated_brier_score(
                y_train, y_train, preds, times
            )
        finally:
            integrated_brier_score_train = np.nan

        try:
            survs = model.predict_survival_function(x_test)
            preds = np.asarray([[fn(t) for t in times] for fn in survs])
            integrated_brier_score_test = integrated_brier_score(
                y_train, y_test, preds, times
            )
        finally:
            integrated_brier_score_test = np.nan

        hl_train = HosmerLemeshowSurvival(
            10, model, x_train, y_train, df=2, Q=10)
        hl_test = HosmerLemeshowSurvival(10, model, x_test, y_test, df=2, Q=10)

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
            "Hosmer-Lemeshow": [f"{hl_train['pvalue']:.2e}", f"{hl_test['pvalue']:.2e}"],
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
