from typing import List, Tuple, Union

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sksurv.metrics import concordance_index_censored
from xgbse.converters import convert_data_to_xgb_format, convert_y


def harrel_c(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> float:
    """
    Compute Harrell's concordance index.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The true target values.
    y_pred : array-like of shape (n_samples,)
        The predicted risk scores.

    Returns
    -------
    float
        The Harrell's concordance index.

    Notes
    -----
    Harrel's concordance index C is defined as the proportion of observations
    that the model can order correctly in terms of survival times.
    """
    risk_score = -y_pred
    event, time = convert_y(y_true)
    return concordance_index_censored(event, time, risk_score)[0]


DEFAULT_PARAMS = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1.20,
    "tree_method": "hist",
    "learning_rate": 5e-2,
    "max_depth": 2,
    "booster": "dart",
}


class XGBSurvival(xgb.Booster):
    def __init__(
        self,
        xgb_params: dict = None,
        num_boost_round: int = 1000,
        n_jobs: int = -1,
        early_stopping_rounds: int = 50
    ) -> None:
        """
        Initialize an XGBSurvival model.

        Parameters
        ----------
        xgb_params : dict, default=None
            XGBoost parameters.
        num_boost_round : int, default=1000
            Number of boosting rounds.
        n_jobs : int, default=-1
            Number of parallel threads used to run XGBoost.
        early_stopping_rounds : int, default=50
            Early stopping rounds for XGBoost.
        """
        if xgb_params is None:
            xgb_params = DEFAULT_PARAMS

        self.xgb_params = xgb_params
        self.n_jobs = n_jobs
        self.feature_importances_ = None
        self.bst = None
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    def fit(
        self,
        x_train: Union[np.ndarray, List],
        y_train: Union[np.ndarray, List],
        validation_data: Tuple[Union[np.ndarray, List], Union[np.ndarray, List]] = None,
        verbose_eval: int = 0,
        callbacks: List = None
    ) -> 'XGBSurvival':
        """
        Fit an XGBSurvival model.

        Parameters
        ----------
        x_train : array-like of shape (n_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples,)
            The target values.
        validation_data : tuple of (array-like, array-like), default=None
            The validation data as a tuple of (x_val, y_val).
        verbose_eval : int, default=0
            Verbosity level.
        callbacks : list, default=None
            Callback list.

        Returns
        -------
        self : XGBSurvival
            Returns self.
        """

        # converting data to xgb format
        dtrain = convert_data_to_xgb_format(
            x_train, y_train, self.xgb_params["objective"])

        # converting validation data to xgb format
        evals = ()
        if validation_data:
            x_val, y_val = validation_data
        else:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=0.1, random_state=42)

        dvalid = convert_data_to_xgb_format(
            x_val, y_val, self.xgb_params["objective"]
        )
        evals = [(dvalid, "validation")]

        # training XGB
        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            early_stopping_rounds=self.early_stopping_rounds,
            evals=evals,
            verbose_eval=verbose_eval,
            callbacks=callbacks,
            num_boost_round=self.num_boost_round,
        )
        self.feature_importances_ = self.bst.get_score()

        return self

    def predict(self, X):
        """
        Make predictions 

        Parameters
        ----------
        X: array-like of shape(n_samples, n_features)
            The input samples.

        Returns
        -------
        preds: array-like of shape(n_samples,)
            The risk scores for each sample.
        """
        # converting to xgb format
        d_matrix = xgb.DMatrix(X)

        # Predictions are the time to event.
        preds = self.bst.predict(d_matrix)

        # Predictions are converted to risk scores :
        preds = -preds
        return preds

    def score(self, X: Union[np.ndarray, List], y: Union[np.ndarray, List]) -> float:
        """
        Evaluate the model on given input data and target values using Harrel's concordance index.

        Parameters
        ----------
        X: array-like of shape(n_samples, n_features)
            The input samples.
        y: array-like of shape(n_samples, 2)
            The target values. The first column contains the event indicator(0 if
            the event has not occurred, 1 otherwise), and the second column contains
            the time to the event.

        Returns
        -------
        score: float
            The Harrel's concordance index score.
        """
        risk_score = self.predict(X)

        scaler = MinMaxScaler()
        risk_score = scaler.fit_transform(risk_score.reshape(-1, 1))
        # The range of this number has to be between 0 and 1, with larger numbers being associated with higher probability of having HF. The values, -Inf, Inf and NA, are not allowed.

        event, time = convert_y(y)
        # Harrel's concordance index C is defined as the proportion of observations
        # that the model can order correctly in terms of survival times.
        return concordance_index_censored(event, time, risk_score)[0]
