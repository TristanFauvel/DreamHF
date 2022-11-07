import xgboost as xgb
from xgbse.converters import convert_data_to_xgb_format, convert_y
from sksurv.metrics import concordance_index_censored
import pandas as pd


def harrel_c(y_true, y_pred):
    """
    Same as concordance_index_censored, but formatted in a way similar to sklearn.metrics.
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

 
class XGBSurvival:
    def __init__(
        self,
        xgb_params=None,
        num_boost_round=1000,
        n_jobs=-1,
    ):
        if xgb_params is None:
            xgb_params = DEFAULT_PARAMS

        self.xgb_params = xgb_params
        self.n_jobs = n_jobs
        self.feature_importances_ = None
        self.bst = None
        self.num_boost_round = num_boost_round

    def fit(
        self,
        X,
        y,
        validation_data=None,
        early_stopping_rounds=None,
        verbose_eval=0,
        callbacks=None,
    ):

        # converting data to xgb format
        dtrain = convert_data_to_xgb_format(X, y, self.xgb_params["objective"])

        # converting validation data to xgb format
        evals = ()
        if validation_data:
            X_val, y_val = validation_data
            dvalid = convert_data_to_xgb_format(
                X_val, y_val, self.xgb_params["objective"]
            )
            evals = [(dvalid, "validation")]

        # training XGB
        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            early_stopping_rounds=early_stopping_rounds,
            evals=evals,
            verbose_eval=verbose_eval,
            callbacks=callbacks,
            num_boost_round=self.num_boost_round,
        )
        self.feature_importances_ = self.bst.get_score()

        return self

    def predict(self, X):
        """
        Predicts survival probabilities using the XGBoost + Logistic Regression pipeline.
        Args:
            X (pd.DataFrame): Dataframe of features to be used as input for the
                XGBoost model.
            return_interval_probs (Bool): Boolean indicating if interval probabilities are
                supposed to be returned. If False the cumulative survival is returned.
                Default is False.
        Returns:
            pd.DataFrame: A dataframe of survival probabilities
            for all times (columns), from a time_bins array, for all samples of X
            (rows). If return_interval_probs is True, the interval probabilities are returned
            instead of the cumulative survival probabilities.
        """

        # converting to xgb format
        d_matrix = xgb.DMatrix(X)

        preds = self.bst.predict(d_matrix)
        return preds

    def score(self, X, y):

        if isinstance(X, pd.DataFrame):
            X = X.values

        risk_score = -self.predict(X)
        event, time = convert_y(y)
        # Harrel's concordance index C is defined as the proportion of observations
        # that the model can order correctly in terms of survival times.

        return concordance_index_censored(event, time, risk_score)[0]

